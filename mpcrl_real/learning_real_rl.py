"""
learning_real.py
----------------
실제 환경용 Learning MPC (시뮬레이션 없이, 논문 기반 greenhouse 물리 모델 사용)
- 센서 입력(x_current, d_current)만으로 동작
- Model.get_true_parameters() 사용
- CasADi / csnlp 0.8.x 버전 호환
"""

from typing import Literal
import numpy as np
import casadi as cs
from csnlp import Nlp
from csnlp.wrappers import Mpc
from mpcrl.util.seeding import RngType
from greenhouse.model_real import Model
from sims.configs.default_real import DefaultReal
import json, os                 # ▼▼ 추가: 경량 RL 훅에 필요한 모듈
from mqtt_handler import get_latest_sensor, publish_actuator


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

        # === 내부 상태/파라미터 캐시 ===
        self._nu = nu
        self._nx = nx
        self._nd = nd
        self._ts = ts
        self._y_min, self._y_max = Model.get_output_range()
        self._y_range = np.asarray(self._y_max - self._y_min, dtype=float).reshape(-1)

        # (라즈베리파이용) 학습 대상 θ 키 정의
        self._theta_keys = [
            "theta_u0","theta_u1","theta_u2",   # 제어 가중치 R
            "theta_w0","theta_w1","theta_w2","theta_w3",  # 제약 위반 가중치 w
            "theta_y",                          # 출력 오차 가중치 Q (스칼라)
            "theta_yfin"                        # 목표 출력 y_fin
        ]


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

    # ---------- RL 관련 ----------
    
    # ---------- θ 벡터 get/set & 저장/로드 유틸 추가 ----------
    def get_theta_keys(self):
        return list(self._theta_keys)

    def get_theta_vector(self, keys=None):
        if keys is None: keys = self._theta_keys
        c_u = np.asarray(self.learnable_pars_init["c_u"], dtype=np.float32).reshape(-1)
        w   = np.asarray(self.learnable_pars_init["w"],   dtype=np.float32).reshape(-1)
        c_y = np.asarray(self.learnable_pars_init["c_y"], dtype=np.float32).reshape(-1)
        y_f = np.asarray(self.learnable_pars_init["y_fin"],dtype=np.float32).reshape(-1)
        full = {
            "theta_u0": c_u[0], "theta_u1": c_u[1], "theta_u2": c_u[2],
            "theta_w0": w[0],   "theta_w1": w[1],   "theta_w2": w[2], "theta_w3": w[3],
            "theta_y":  c_y[0],
            "theta_yfin": y_f[0],
        }
        return np.array([full[k] for k in keys], dtype=np.float32)

    def set_theta_vector(self, vec, keys=None):
        if keys is None: keys = self._theta_keys
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        assert len(vec) == len(keys)
        kv = dict(zip(keys, vec.tolist()))
        self.learnable_pars_init["c_u"]   = np.array([kv["theta_u0"], kv["theta_u1"], kv["theta_u2"]], dtype=float)
        self.learnable_pars_init["w"]     = np.array([kv["theta_w0"], kv["theta_w1"], kv["theta_w2"], kv["theta_w3"]], dtype=float)
        self.learnable_pars_init["c_y"]   = np.array([kv["theta_y"]], dtype=float)
        self.learnable_pars_init["y_fin"] = np.array([kv["theta_yfin"]], dtype=float)

    def save_theta_json(self, path, keys=None, vec=None):
        if vec is None: vec = self.get_theta_vector(keys)
        if keys is None: keys = self.get_theta_keys()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({k: float(v) for k, v in zip(keys, vec)}, f, indent=2)

    def load_theta_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        keys = self.get_theta_keys()
        vec = np.array([data[k] for k in keys], dtype=np.float32)
        self.set_theta_vector(vec, keys)

    # ---------- 스테이지 리워드(논문 L) 계산 함수 추가 ----------
    def compute_stage_reward(self, y_scalar, u_vec, du_vec=None, slack_vec=None):
        # L = (y - y_fin)^2 * c_y + u^2 * c_u + (Δu)^2 * S + (w·s / range)
        c_u = np.asarray(self.learnable_pars_init["c_u"], dtype=float).reshape(-1)
        c_y = float(np.asarray(self.learnable_pars_init["c_y"], dtype=float).reshape(-1)[0])
        y_f = float(np.asarray(self.learnable_pars_init["y_fin"], dtype=float).reshape(-1)[0])
        w   = np.asarray(self.learnable_pars_init["w"], dtype=float).reshape(-1)

        ly = c_y * float((y_scalar - y_f) ** 2)
        lu = float(np.sum(c_u * (np.asarray(u_vec, dtype=float) ** 2)))
        ldu = 0.0
        if du_vec is not None:
            ldu = 0.1 * float(np.sum((np.asarray(du_vec, dtype=float) ** 2)))
        lsl = 0.0
        if slack_vec is not None:
            denom = float(self._y_range[0] if self._y_range[0] != 0 else 1.0)
            lsl = float(np.dot(w, np.asarray(slack_vec, dtype=float) / denom))

        # 보상은 "높을수록 좋음" → 비용의 음수 반환
        return -(ly + lu + ldu + lsl)

    # ---------- V(s) 평가(정책문제 목적값) 함수 추가 ----------
    def eval_V(self, x_vec, d_vec=None):
        pars = {}
        for k, v in self.learnable_pars_init.items():
            pars[k] = v
        for k, v in self.fixed_pars.items():
            pars[k] = v
        if d_vec is None:
            d_vec = np.zeros((self._nd,), dtype=float)
        pars["d"] = np.tile(np.asarray(d_vec, dtype=float).reshape(-1, 1), (1, self.prediction_horizon))
        pars["x_0"] = np.asarray(x_vec, dtype=float).reshape(-1, 1)
        try:
            sol = self.solve(pars=pars, vals0={"x": np.asarray(x_vec, dtype=float).reshape(-1, 1)})
            return float(sol.obj)  # 작을수록 좋음 (cost)
        except Exception:
            return float("+inf")

    # ---------- Q(s,a) 평가(1-step rollout + γV) 함수 추가 ----------
    def eval_Q(self, x_vec, d_vec, a0_vec):
        x0 = np.asarray(x_vec, dtype=float).reshape(-1, 1)
        u0 = np.asarray(a0_vec, dtype=float).reshape(-1, 1)
        d0 = np.asarray(d_vec, dtype=float).reshape(-1, 1)

        p_values = Model.get_true_parameters()
        p = cs.DM.vertcat(*[v for v in p_values])

        # 1-step 모델 전개
        x1_cs = Model.rk4_step(cs.DM(x0), cs.DM(u0), cs.DM(d0), p, self._ts)
        x1 = np.array(x1_cs).reshape(-1, 1)

        # 출력 y1 계산 (슬랙은 경량 근사로 생략)
        y1 = np.array(Model.output(cs.DM(x1), p)).reshape(-1, 1)
        y_scalar = float(y1[0, 0])

        # 스테이지 비용 (보상은 음수)
        stage_cost = -self.compute_stage_reward(y_scalar, u0.flatten(), du_vec=None, slack_vec=None)

        # 다음 상태 가치 V(x1)
        v_x1 = self.eval_V(x1.flatten(), d_vec)

        return float(stage_cost + self.discount_factor * v_x1)

    # ---------- 제약 위반 지표(Ψ) 계산 함수 추가 ----------
    def compute_violation(self, y_vec):
        y_min, y_max = self._y_min, self._y_max
        below = np.maximum(0.0, (y_min - y_vec) / np.maximum(1e-6, (y_max - y_min)))
        above = np.maximum(0.0, (y_vec - y_max) / np.maximum(1e-6, (y_max - y_min)))
        return float(np.sum(below + above))

    # ---------- 센서 입출력 훅(임시 스텁) & 제어 step 함수 추가 ----------
    def read_current_measurements(self):
        data = get_latest_sensor()
        if not data:
            print("⏳ MQTT 센서 데이터 대기 중...")
            return np.zeros((self._nx,)), np.zeros((1,)), np.zeros((1,)), np.zeros((self._nd,))
        try:
            temp = float(data.get("temp", 0))
            hum  = float(data.get("hum", 0))
            co2  = float(data.get("co2", 0))
            light= float(data.get("light", 0))
            x = np.array([0.0, co2, temp, hum], dtype=float)
            y = np.array([temp], dtype=float)
            d = np.array([light, co2, temp, hum], dtype=float)
            return x, y, y, d
        except Exception as e:
            print(f"[Sensor Parse Error] {e}")
            return np.zeros((self._nx,)), np.zeros((1,)), np.zeros((1,)), np.zeros((self._nd,))


    def read_next_measurements(self):
        return self.read_current_measurements() 

    def apply_actuators(self, u_vec):
        try:
            publish_actuator(u_vec)
        except Exception as e:
            print(f"[MQTT Publish Error] {e}")


    def control_step(self, x_vec, d_vec):
        pars = {}
        for k, v in self.learnable_pars_init.items():
            pars[k] = v
        for k, v in self.fixed_pars.items():
            pars[k] = v
        pars["d"] = np.tile(np.asarray(d_vec, dtype=float).reshape(-1, 1), (1, self.prediction_horizon))
        pars["x_0"] = np.asarray(x_vec, dtype=float).reshape(-1, 1)
        try:
            sol = self.solve(pars=pars, vals0={"x": np.asarray(x_vec, dtype=float).reshape(-1, 1)})
            u0 = np.array(sol.vals["u"][:, 0]).flatten()
            return u0.astype(np.float32), sol.status
        except Exception as e:
            print(f"[SolverError-control_step] {e}")
            return np.zeros((self._nu,), dtype=np.float32), "Solve_Failed"
            

