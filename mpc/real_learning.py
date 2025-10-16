#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Literal
import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc

from greenhouse.model import Model

class RealLearningMpc(Mpc[cs.SX]):
    """Real sensor-based MPC for greenhouse control."""

    def __init__(
        self,
        greenhouse_env=None,
        test=None,
        prediction_horizon: int = 6*4,
        prediction_model: Literal["euler", "rk4"] = "rk4",
        constrain_control_rate: bool = True,
    ):
        # --------------------------
        # 실제 환경용: 차원 하드코딩
        self.nx = 4  # temp, hum, co2, light
        self.nu = 4  # heater, humidifier, co2_valve, led
        self.nd = 0
        self.ts = 1.0

        # discount factor
        if test is not None:
            self.discount_factor = test.discount_factor
        else:
            self.discount_factor = 0.99

        # --------------------------
        # Nlp 기반 초기화
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, prediction_horizon=prediction_horizon)
        N = prediction_horizon

        # dynamics parameter은 perturb 없이 기본값
        self.learnable_pars_init = {}
        self.fixed_pars = {}

        # --------------------------
        # 상태, 액션 변수
        x, _ = self.state("x", self.nx, lb=0, ub=1e3)
        u, _ = self.action("u", self.nu, lb=Model.get_u_min().reshape(-1,1),
                            ub=Model.get_u_max().reshape(-1,1))
        s, _, _ = self.variable("s", (self.nx, N+1), lb=0)  # slack

        # dynamics
        if prediction_model == "euler":
            model = lambda x,u,d: Model.euler_step(x,u,d,None,self.ts)
        else:
            model = lambda x,u,d: Model.rk4_step(x,u,d,None,self.ts)

        self.set_dynamics(lambda x,u,d: model(x,u,d), n_in=3, n_out=1)

        # output constraints (dummy)
        y = [x[:,k] for k in range(N+1)]
        for k in range(N+1):
            self.constraint(f"y_min_{k}", y[k], ">=", 0)
            self.constraint(f"y_max_{k}", y[k], "<=", 1e3)

        # control rate constraint
        if constrain_control_rate:
            du_lim = Model.get_du_lim()
            for k in range(1,N):
                self.constraint(f"du_min_{k}", u[:,k]-u[:,k-1], "<=", du_lim)
                self.constraint(f"du_max_{k}", u[:,k]-u[:,k-1], ">=", -du_lim)

        # objective: dummy, 실제 센서 값 기반 MPC에서 step()에서 수정 가능
        self.minimize(0)

        # solver
        opts = {"expand": True,
                "show_eval_warnings": False,
                "ipopt":{"print_level":0}}
        self.init_solver(opts, solver="ipopt")

    def step(self, x_state):
        """실제 센서 값 x_state를 받아서 액추에이터 명령 반환"""
        # 여기서는 단순 proportional dummy 제어 예시
        # heater, humidifier, co2_valve, led
        target = np.array([21, 65, 650, 325])
        u = np.clip((target - np.array(x_state))*0.1, 0, 1)  # 단순 비례 제어
        return u.tolist()

    def update_theta(self, new_theta):
        """Fine-tune theta 업데이트"""
        self.learnable_pars_init.update(new_theta)
