"""
learning_real.py
----------------
실제 환경용 Learning MPC (시뮬레이션 없이, 논문 기반 greenhouse 물리 모델 사용)
- 센서 입력(x_current, d_current)만으로 동작
- Model.get_true_parameters() 사용
- CasADi / csnlp 0.8.x 버전 호환

ONLY MPC 버전으로  learning_real_rl.py 충돌날 시 롤백 요망
"""

from typing import Literal
import numpy as np
import casadi as cs
from csnlp import Nlp
from csnlp.wrappers import Mpc
from mpcrl.util.seeding import RngType
from greenhouse.model_real import Model
from sims.configs.default_real import DefaultReal


class LearningMpcReal(Mpc[cs.SX]):
    """Real-world MPC controller for greenhouse control."""

    def __init__(
        self,
        nx: int = 4,
        nu: int = 3,
        nd: int = 4,
        ts: float = 60.0 * 15.0,  # 15분 주기
        test: DefaultReal | None = None,
        np_random: RngType | None = None,
        prediction_horizon: int = 6 * 4,  # 6시간 (15분 단위 step)
        prediction_model: Literal["euler", "rk4"] = "rk4",
        constrain_control_rate: bool = True,
    ):
        # ---------- 기본 설정 ----------
        test = test or DefaultReal()
        np_random = np.random.default_rng(np_random)
        N = prediction_horizon

        # ---------- Model 파라미터 ----------
        u_min, u_max, du_lim, y_range = (
            Model.get_u_min(),
            Model.get_u_max(),
            Model.get_du_lim(),
            Model.get_output_range(),
        )

        # ---------- NLP 초기화 ----------
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, prediction_horizon=N)
        self.discount_factor = test.discount_factor

        # ---------- 파라미터 등록 ----------
        # 학습 가능한 cost 관련 파라미터
        V0 = self.parameter("V0", (1,))
        c_u = self.parameter("c_u", (nu,))
        c_dy = self.parameter("c_dy", (1,))
        c_y = self.parameter("c_y", (1,))
        y_fin = self.parameter("y_fin", (1,))
        w = self.parameter("w", (4,))
        olb = self.parameter("olb", (4,))
        oub = self.parameter("oub", (4,))

        # ---------- 논문 기반 물리 파라미터 ----------
        p = [self.parameter(f"p_{i}", (1,)) for i in range(Model.n_params)]
        p_values = Model.get_true_parameters()

        learnable_pars_init = test.learnable_pars_init
        fixed_pars = test.fixed_pars

        # 모든 물리 파라미터를 고정 파라미터로 등록
        for i in range(Model.n_params):
            fixed_pars[f"p_{i}"] = np.asarray(p_values[i])

        self.learnable_pars_init = learnable_pars_init
        self.fixed_pars = fixed_pars
        p = cs.vertcat(*p)

        # ---------- 변수 정의 ----------
        x, _ = self.state("x", nx, lb=0, ub=1e3)
        u, _ = self.action("u", nu, lb=u_min.reshape(-1, 1), ub=u_max.reshape(-1, 1))
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N + 1), lb=0)
        self.f_dynamics = (
            lambda x, u, d: Model.rk4_step(x, u, d, p, ts)
            if prediction_model == "rk4"
            else Model.euler_step(x, u, d, p, ts)
        )

        # ---------- 제약조건 ----------
        y = [Model.output(x[:, k], p) for k in range(N + 1)]
        # ---------- 초기 상태 제약 ----------
        self.constraint("x_init", x[:, 0], "==", self.parameters["x_0"])

        # ✅ 수정된 부분
        y_min, y_max = Model.get_output_range()
        y_range = cs.DM(y_max - y_min)

        for k in range(N + 1):
            self.constraint(f"y_min_{k}", y[k], ">=", (1 + olb) * y_min - s[:, k])
            self.constraint(f"y_max_{k}", y[k], "<=", (1 + oub) * y_max + s[:, k])

        if constrain_control_rate:
            for k in range(1, N):
                self.constraint(f"du_min_{k}", u[:, k] - u[:, k - 1], "<=", du_lim)
                self.constraint(f"du_max_{k}", u[:, k] - u[:, k - 1], ">=", -du_lim)

        # ---------- 목적함수 (논문식 제곱 오차 기반) ----------
        obj = V0

        # 상태 추종 목표 (y_ref)
        y_ref = y_fin  # (shape: (1,)) — 온도 기반 목표
        y_min, y_max = Model.get_output_range()
        y_range = cs.DM(y_max - y_min)

        # 제어 변화율 penalty 계수 (논문식 S)
        c_du = 0.1 * cs.DM.ones(nu)

        for k in range(N + 1):
            # (1) 상태 오차 비용: (y - y_ref)^2 * Q
            err_y = y[k][0, 0] - y_ref[0]
            obj += (self.discount_factor**k) * c_y[0] * cs.sumsqr(err_y)

            # (2) 슬랙 변수 비용 (제약 완화 penalty)
            obj += (self.discount_factor**k) * cs.dot(w, s[:, k] / y_range)

        for k in range(N):
            # (3) 제어 입력 크기 비용: u^2 * R
            obj += (self.discount_factor**k) * cs.dot(c_u, cs.sumsqr(u[:, k]))

            # (4) 제어 변화율 비용: (Δu)^2 * S
            if k > 0:
                obj += (self.discount_factor**k) * cs.dot(c_du, cs.sumsqr(u[:, k] - u[:, k-1]))

        self.minimize(obj)

        # ---------- Solver 설정 ----------
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "sb": "yes",
                "print_level": 0,
                #"max_iter": 2000,
                "max_iter": 500,           # ✅ 2000 → 500 으로 줄이기 (속도 ↑)
                "tol": 1e-4,               # ✅ 허용 오차 완화 (속도 ↑)
                "linear_solver": "mumps",
                "acceptable_tol": 1e-3,
            },
        }
        self.init_solver(opts, solver="ipopt")

    # ---------- 실시간 제어 입력 계산 ----------
    def compute_control(self, x_current: np.ndarray, d_current: np.ndarray):
        """센서 입력(x_current, d_current)을 받아 최적 제어 입력 계산"""
        params = {}

        # 1️⃣ 학습 가능한 파라미터 초기값
        for k, v in self.learnable_pars_init.items():
            params[k] = v

        # 2️⃣ 고정 파라미터
        for k, v in self.fixed_pars.items():
            params[k] = v

        # 3️⃣ 외란 입력
        params["d"] = np.tile(d_current.reshape(-1, 1), (1, self.prediction_horizon))

        # 4️⃣ 초기 상태
        params["x_0"] = x_current.reshape(-1, 1)

        try:
            sol = self.solve(pars=params, vals0={"x": x_current.reshape(-1, 1)})
            u_opt = np.array(sol.vals["u"][:, 0]).flatten()
            return u_opt, sol.status
        except Exception as e:
            print(f"[SolverError] {e}")
            return np.zeros((3,)), "Solve_Failed"
