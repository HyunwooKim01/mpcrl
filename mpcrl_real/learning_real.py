"""
learning_real.py
----------------
실제 환경(센서 입력 기반)에서 동작하는 LearningMpc 제어기 버전.
시뮬레이션 환경 없이, Model 기반 예측만 수행.
"""

from typing import Literal
import numpy as np
import casadi as cs
from csnlp import Nlp
from csnlp.wrappers import Mpc
from mpcrl.util.seeding import RngType

from greenhouse.model import Model
from sims.configs.default import DefaultTest


class LearningMpcReal(Mpc[cs.SX]):
    """Real-world MPC controller for greenhouse control.
    (No simulation environment)
    """

    def __init__(
        self,
        nx: int = 4,
        nu: int = 3,
        nd: int = 4,
        ts: float = 60.0 * 15.0,  # 15 min sampling
        test: DefaultTest | None = None,
        np_random: RngType | None = None,
        prediction_horizon: int = 6 * 4,  # 6 hours
        prediction_model: Literal["euler", "rk4"] = "rk4",
        constrain_control_rate: bool = True,
    ):
        # 기본 설정
        test = test or DefaultTest()
        np_random = np.random.default_rng(np_random)
        N = prediction_horizon

        # Model 파라미터 및 제약 조건
        u_min, u_max, du_lim, y_range = (
            Model.get_u_min(),
            Model.get_u_max(),
            Model.get_du_lim(),
            Model.get_output_range(),
        )

        # NLP 초기화
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, prediction_horizon=N)
        self.discount_factor = test.discount_factor

        # -------------- Parameters --------------
        # MPC 학습 가능한 파라미터 (기존 RL-MPC에서도 동일)
        V0 = self.parameter("V0", (1,))
        c_u = self.parameter("c_u", (nu,))
        c_dy = self.parameter("c_dy", (1,))
        c_y = self.parameter("c_y", (1,))
        y_fin = self.parameter("y_fin", (1,))
        w = self.parameter("w", (4,))
        olb = self.parameter("olb", (4,))
        oub = self.parameter("oub", (4,))

        # dynamics parameters
        p = [self.parameter(f"p_{i}", (1,)) for i in range(Model.n_params)]
        p_values = Model.get_perturbed_parameters(test.p_perturb, np_random=np_random)
        learnable_pars_init = test.learnable_pars_init
        fixed_pars = test.fixed_pars

        for i in range(Model.n_params):
            if i in test.p_learn:
                learnable_pars_init[f"p_{i}"] = np.asarray(p_values[i])
            else:
                fixed_pars[f"p_{i}"] = np.asarray(p_values[i])

        self.learnable_pars_init = learnable_pars_init
        self.fixed_pars = fixed_pars
        p = cs.vertcat(*p)

        # -------------- Variables --------------
        x, _ = self.state("x", nx, lb=0, ub=1e3)
        u, _ = self.action("u", nu, lb=u_min.reshape(-1, 1), ub=u_max.reshape(-1, 1))
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N + 1), lb=0)  # slack

        # -------------- Model Dynamics --------------
        if prediction_model == "euler":
            model = lambda x, u, d: Model.euler_step(x, u, d, p, ts)
        else:
            model = lambda x, u, d: Model.rk4_step(x, u, d, p, ts)
        self.set_dynamics(lambda x, u, d: model(x, u, d), n_in=3, n_out=1)

        # -------------- Constraints --------------
        y = [Model.output(x[:, k], p) for k in range(N + 1)]
        y_min = Model.get_output_min(np.zeros((nd,)))
        y_max = Model.get_output_max(np.zeros((nd,)))

        for k in range(N + 1):
            self.constraint(f"y_min_{k}", y[k], ">=", (1 + olb) * y_min - s[:, k])
            self.constraint(f"y_max_{k}", y[k], "<=", (1 + oub) * y_max + s[:, k])

        if constrain_control_rate:
            for k in range(1, N):
                self.constraint(f"du_min_{k}", u[:, k] - u[:, k - 1], "<=", du_lim)
                self.constraint(f"du_max_{k}", u[:, k] - u[:, k - 1], ">=", -du_lim)

        # -------------- Objective Function --------------
        obj = V0
        for k in range(N):
            for j in range(nu):
                obj += (self.discount_factor**k) * c_u[j] * u[j, k]
        for k in range(N + 1):
            obj += (self.discount_factor**k) * cs.dot(w, s[:, k] / y_range)
        for k in range(1, N + 1):
            obj += -(self.discount_factor**k) * c_dy * (y[k][0] - y[k - 1][0])
        obj += (self.discount_factor ** (N + 1)) * c_dy * c_y * (y_fin - y[N][0])
        self.minimize(obj)

        # -------------- Solver Options --------------
        opts = {
            "expand": True,
            "show_eval_warnings": False,
            "print_time": False,
            "ipopt": {
                "sb": "yes",
                "print_level": 0,
                "max_iter": 2000,
                "linear_solver": "ma57",
            },
        }
        self.init_solver(opts, solver="ipopt")

    # -------------- Real-time Step Interface --------------
    def compute_control(self, x_current: np.ndarray, d_current: np.ndarray):
        """실제 센서 입력(x_current, d_current)을 받아 최적 제어 입력 계산"""
        params = self.fixed_pars.copy()
        params["d"] = np.tile(d_current.reshape(-1, 1), (1, self.prediction_horizon))
        sol = self.solve(pars=params, vals0={"x": x_current.reshape(-1, 1)})
        u_opt = np.array(sol.vals["u"][:, 0]).flatten()
        return u_opt, sol.status
