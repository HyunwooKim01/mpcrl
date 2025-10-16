#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Literal
import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from greenhouse.model import Model


class RealLearningMpc(Mpc[cs.SX]):
    """Real-sensor based MPC for greenhouse control."""

    def __init__(
        self,
        greenhouse_env=None,
        test=None,
        prediction_horizon: int = 6*4,
        prediction_model: Literal["euler", "rk4"] = "rk4",
        constrain_control_rate: bool = True,
    ):
        # 실제 환경용 상태/액션 차원
        self._nx = 4
        self._nu = 4
        self._nd = 0
        self._ts = 1.0

        # discount factor
        self.discount_factor = 0.99 if test is None else test.discount_factor

        # Nlp 기반 초기화
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, prediction_horizon=prediction_horizon)

        # MPC 변수 정의
        x, _ = self.state("x", self._nx, lb=0, ub=1e3)
        u, _ = self.action(
            "u",
            self._nu,
            lb=Model.get_u_min().reshape(-1, 1),
            ub=Model.get_u_max().reshape(-1, 1),
        )

        # slack variables
        s, _, _ = self.variable("s", (self._nx, prediction_horizon + 1), lb=0)

        # disturbance placeholder
        d = cs.MX.sym("d", self._nd) if self._nd > 0 else cs.MX.zeros(self._nd, 1)

        # dynamics 함수
        if prediction_model == "euler":
            model = lambda x, u, d: Model.euler_step(x, u, d, None, self._ts)
        else:
            model = lambda x, u, d: Model.rk4_step(x, u, d, None, self._ts)
        self.set_dynamics(lambda x, u, d: model(x, u, d), n_in=3, n_out=1)

        # objective: 단순 예시 (heater, humidifier, CO2, led 최소화)
        obj = 0
        for k in range(prediction_horizon):
            obj += cs.dot(u[:, k], u[:, k])
        self.minimize(obj)

        # solver 설정
        opts = {
            "expand": True,
            "show_eval_warnings": False,
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "linear_solver": "mumps",
                "hessian_approximation": "limited-memory",
                "tol": 1e-5,
                "acceptable_tol": 1e-2,
                "max_iter": 2000,
            },
        }
        self.init_solver(opts, solver="ipopt")

    def step(self, x_current):
        """실제 센서 값 기반 MPC 제어"""
        # CasADi 변수에 넣기
        x_val = np.array(x_current).reshape((self._nx, 1))
        # disturbance는 0으로 처리
        d_val = np.zeros((self._nd, 1))
        u_opt = super().step(x_val, d_val)
        return np.array(u_opt).flatten()
