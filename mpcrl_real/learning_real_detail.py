# learning_real_detail.py
# ✅ CasADi 기반 MPC + RL 파라미터 로드 기능 (논문식 (18)~(21) 대응)
from __future__ import annotations
import time
import numpy as np
import casadi as cs
import pickle


class LearningMpcCasADi:
    """CasADi 기반 MPC (Raspberry Pi 실시간용)
    상태 x=[temp_in, hum_in, co2_in, light_in]
    제어 u=[fan, heater, led]
    외란 d=[solar_rad, co2_out, temp_out, hum_out]
    논문식 (18)~(21) 기반 + RL 파라미터 적용 구조
    """

    def __init__(
        self,
        ts: float = 5.0,
        N: int = 24,
        du_lim=(0.05, 0.05, 0.05),
        u_min=(0.0, 0.0, 0.0),
        u_max=(1.0, 1.0, 1.0),
        Q=(2.0, 1.0, 0.2, 0.5),
        R=(0.05, 0.1, 0.1),
        S=(0.2, 0.3, 0.3),
        alpha_growth: float = 1.0,
    ):
        self.ts = ts
        self.N = N
        self._nx, self._nu = 4, 3
        self.u_prev = np.zeros(3)
        self.r = np.array([20.0, 60.0, 420.0, 300.0])

        # MPC 파라미터 초기값
        self.theta_dy1 = 100.0
        self.theta_u_vec = np.array([10.0, 1.0, 1.0])
        self.theta_w_vec = np.array([1e5, 1e5])
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.S = np.diag(S)
        self.alpha_growth = alpha_growth
        self.theta_omega = np.array([5.0, 5.0, 0.0, 0.0])  # 슬랙 가중치
        self.theta_y1f = 2.0  # 터미널 온도항 가중치


        # MPC 초기화
        self._build_mpc()

    # ─────────────────────────────────────────────
    # MPC 모델 구성 (CasADi 기반)
    # ─────────────────────────────────────────────
    def _build_mpc(self):
        nx, nu, nd, N = self._nx, self._nu, 4, self.N
        x = cs.MX.sym("x", nx, N + 1)
        u = cs.MX.sym("u", nu, N)
        sigma = cs.MX.sym("sigma", nx, N)  # 슬랙 변수 σ(k)
        du = cs.MX.sym("du", nu, N)
        x0 = cs.MX.sym("x0", nx)
        d = cs.MX.sym("d", nd)
        u_prev = cs.MX.sym("u_prev", nu)
        r = cs.MX.sym("r", nx)

        # --- Dynamics ---
        def f_dyn(xk, uk, dk):
            # ✅ MX 타입은 언패킹 불가 → 인덱싱으로 접근
            temp = xk[0]
            hum = xk[1]
            co2 = xk[2]
            light = xk[3]

            fan = uk[0]
            heater = uk[1]
            led = uk[2]

            rad = dk[0]
            co2_out = dk[1]
            temp_out = dk[2]
            hum_out = dk[3]

            dtemp = 0.015 * (temp_out - temp) + 0.1 * heater - 0.07 * fan
            dhum = 0.01 * (hum_out - hum) - 0.06 * fan + 0.015 * heater
            dco2 = 0.002 * (co2_out - co2)
            dlight = -0.05 * light + 1.2 * led * 300.0 + 0.15 * rad
            return cs.vertcat(dtemp, dhum, dco2, dlight)

        g = []
        J = 0
        g.append(x[:, 0] - x0)

        for k in range(N):
            x_next = x[:, k] + self.ts * f_dyn(x[:, k], u[:, k], d)
            g.append(x[:, k + 1] - x_next)

            if k == 0:
                g.append(du[:, k] - (u[:, k] - u_prev))
            else:
                g.append(du[:, k] - (u[:, k] - u[:, k - 1]))

            # --- 논문식 (18)~(21) 대응 ---
            J_track = cs.mtimes([(x[:, k] - r).T, self.Q, (x[:, k] - r)])
            J_du = cs.mtimes([du[:, k].T, self.R, du[:, k]])
            J_energy = cs.mtimes([u[:, k].T, self.S, u[:, k]])

            T = x[0, k]
            H = x[1, k]
            L = x[3, k]
            Tmin, Tmax = 18.0, 28.0
            Hmin, Hmax = 40.0, 80.0
            # σ 제약식 추가: y_min - σ ≤ y ≤ y_max + σ
            y_min = np.array([18.0, 40.0, 300.0, 0.0])
            y_max = np.array([28.0, 80.0, 1000.0, 1.0])
            g.append(x[:, k] - (y_max + sigma[:, k]))
            g.append((y_min - sigma[:, k]) - x[:, k])

            # Slack penalty (논문식 θ_ωᵀσ⊙y_range)
            y_range = np.array([1.6, 5.0, 70.0, 1.0])
            J_slack = cs.dot(self.theta_omega, sigma[:, k] * y_range)

            # Growth proxy + Δgrowth
            gT = cs.exp(-((T - 25.0) ** 2) / (2 * 3 ** 2))
            gH = cs.exp(-((H - 70.0) ** 2) / (2 * 10 ** 2))
            gL = cs.exp(-((L - 300.0) ** 2) / (2 * 100.0 ** 2))
            growth = gT * gH * gL

            if k == 0:
                delta_growth = growth
            else:
                prev_growth = cs.exp(-((x[0, k-1] - 25.0) ** 2) / (2 * 3 ** 2)) * \
                            cs.exp(-((x[1, k-1] - 70.0) ** 2) / (2 * 10 ** 2)) * \
                            cs.exp(-((x[3, k-1] - 300.0) ** 2) / (2 * 100.0 ** 2))
                delta_growth = growth - prev_growth

            # 통합 비용 (논문식 22a 근사)
            J += J_track + J_du + J_energy + J_slack - self.alpha_growth * delta_growth

        # terminal cost (논문식 θ_{y1,f}(y1(N)-r1)^2)
        J += self.theta_y1f * cs.sumsqr(x[0, N] - r[0])

        w = cs.vertcat(
            cs.reshape(x, -1, 1),
            cs.reshape(u, -1, 1),
            cs.reshape(du, -1, 1),
            cs.reshape(sigma, -1, 1),
        )

        g = cs.vertcat(*g)
        p = cs.vertcat(x0, d, u_prev, r)

        nlp = {"x": w, "f": J, "g": g, "p": p}
        opts = {
            "qpsol": "qrqp",
            "max_iter": 40,
            "print_time": False,
            "print_header": False,
            "print_iteration": False,
        }
        self.solver = cs.nlpsol("solver", "sqpmethod", nlp, opts)
        self.w0 = np.zeros(((nx * (N + 1)) + (nu * N * 2) + (nx * N), 1))

    # ─────────────────────────────────────────────
    # RL(Q-learning) 파라미터 로드
    # ─────────────────────────────────────────────
    def load_theta(self, path: str):
        """RL에서 학습된 θ 파라미터(Q,R,S,α)를 로드"""
        try:
            with open(path, "rb") as f:
                theta = pickle.load(f)
            if "Q" in theta:
                self.Q = np.diag(theta["Q"])
            if "R" in theta:
                self.R = np.diag(theta["R"])
            if "S" in theta:
                self.S = np.diag(theta["S"])
            if "alpha_growth" in theta:
                self.alpha_growth = theta["alpha_growth"]
            print(f"✅ Loaded RL parameters from {path}")
        except Exception as e:
            print(f"⚠️ Failed to load RL parameters: {e}")

    # ─────────────────────────────────────────────
    # 참조(목표값) 설정
    # ─────────────────────────────────────────────
    def set_reference(self, Tmid, Hmid, CO2_ref=420.0, L_ref=300.0):
        self.r = np.array([Tmid, Hmid, CO2_ref, L_ref])
        self.T_ref, self.H_ref = Tmid, Hmid

    # ─────────────────────────────────────────────
    # MPC policy: 제어 입력만 반환 (reward는 RL 계산)
    # ─────────────────────────────────────────────
    def policy(self, s: np.ndarray):
        x = np.array(s[:4])
        d = np.array(s[4:8])
        p = np.concatenate([x, d, self.u_prev, self.r]).reshape(-1, 1)
        nx, nu, N = self._nx, self._nu, self.N
        n_eq = (N + 1) * nx + N * nu

        try:
            sol = self.solver(
                x0=self.w0, p=p,
                lbg=np.zeros((n_eq, 1)),
                ubg=np.zeros((n_eq, 1)),
                lbx=-1e9, ubx=+1e9
            )
        except Exception as e:
            print(f"[WARN] CasADi solver failed: {e}")
            sol = {"x": np.zeros_like(self.w0)}

        w_opt = np.array(sol["x"]).flatten()
        u_opt = w_opt[(N + 1) * nx:(N + 1) * nx + nu]
        u_opt = np.clip(u_opt, 0.0, 1.0)
        self.u_prev = u_opt.copy()
        self.w0 = w_opt.reshape(-1, 1)

        print(f"⚙️ u={u_opt.round(3)}")
        return u_opt
