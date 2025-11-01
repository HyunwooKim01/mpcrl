# learning_real_detail.py
# ✅ CasADi 기반 MPC + RL 파라미터 로드 기능 (논문식 (18)~(21), (22b)~(22h) 대응)
from __future__ import annotations
import numpy as np
import casadi as cs
import pickle


class LearningMpcCasADi:
    """CasADi 기반 MPC (Raspberry Pi 실시간용)
    상태 x=[temp_in, hum_in, co2_in, light_in]
    제어 u=[fan, heater, led]
    외란 d=[solar_rad, co2_out, temp_out, hum_out]
    논문식 (18)~(21) + (22b)~(22h) 기반 + RL 파라미터 적용 구조
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
        self.r = np.array([20.0, 60.0, 420.0, 300.0])  # [T, H, CO2, L] 참조

        # 제약 파라미터
        self.u_min = np.array(u_min, dtype=float)
        self.u_max = np.array(u_max, dtype=float)
        self.du_lim = np.array(du_lim, dtype=float)

        # MPC 가중치 (RL이 덮어씀)
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.S = np.diag(S)
        self.alpha_growth = alpha_growth

        # slack/terminal 가중
        self.theta_omega = np.array([5.0, 5.0, 0.0, 0.0])  # σ 가중치(온/습 위주)
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
        sigma = cs.MX.sym("sigma", nx, N)
        du = cs.MX.sym("du", nu, N)
        x0 = cs.MX.sym("x0", nx)
        d = cs.MX.sym("d", nd)
        u_prev = cs.MX.sym("u_prev", nu)
        r = cs.MX.sym("r", nx)

        # --- 선형/준선형 동역학 근사 (실환경용) ---
        def f_dyn(xk, uk, dk):
            temp, hum, co2, light = xk[0], xk[1], xk[2], xk[3]
            fan, heater, led = uk[0], uk[1], uk[2]
            rad, co2_out, temp_out, hum_out = dk[0], dk[1], dk[2], dk[3]
            dtemp = 0.015 * (temp_out - temp) + 0.1 * heater - 0.07 * fan
            dhum = 0.01 * (hum_out - hum) - 0.06 * fan + 0.015 * heater
            dco2 = 0.002 * (co2_out - co2)
            dlight = -0.05 * light + 1.2 * led * 300.0 + 0.15 * rad
            return cs.vertcat(dtemp, dhum, dco2, dlight)

        # 제약식/경계, 비용 초기화
        g, lbg, ubg = [], [], []
        J = 0

        # (22b) 초기 상태
        g.append(x[:, 0] - x0)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        # 출력 제한 값 (작물 범위)
        y_min = np.array([18.0, 40.0, 300.0, 0.0])
        y_max = np.array([28.0, 80.0, 1000.0, 1.0])
        y_range = np.array([1.6, 5.0, 70.0, 1.0])  # slack 스케일링

        # Horizon 루프
        for k in range(N):
            # (22c) 시스템 동역학
            x_next = x[:, k] + self.ts * f_dyn(x[:, k], u[:, k], d)
            g.append(x[:, k + 1] - x_next)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

            # (22e) Δu 제약 (정의식: du = u(k)-u(k-1 or prev))
            if k == 0:
                du_k = u[:, k] - u_prev
            else:
                du_k = u[:, k] - u[:, k - 1]
            g.append(du[:, k] - du_k)
            lbg += [0.0] * nu
            ubg += [0.0] * nu

            # (22e 확장) Δu 한계: -du_lim ≤ du ≤ du_lim
            g.append(du[:, k])
            lbg += list(-self.du_lim)
            ubg += list(self.du_lim)

            # 입력 범위: u_min ≤ u ≤ u_max
            g.append(u[:, k])
            lbg += list(self.u_min)
            ubg += list(self.u_max)

            # (22g) 출력 제약 (soft): y - (ymax + σ) ≤ 0, (ymin - σ) - y ≤ 0
            g.append(x[:, k] - (y_max + sigma[:, k]))
            g.append((y_min - sigma[:, k]) - x[:, k])
            lbg += [-cs.inf] * nx * 2
            ubg += [0.0] * nx * 2

            # 비용항 (22a)
            e = x[:, k] - r
            J_track = cs.mtimes([e.T, self.Q, e])
            J_du = cs.mtimes([du[:, k].T, self.R, du[:, k]])
            J_energy = cs.mtimes([u[:, k].T, self.S, u[:, k]])
            J_slack = cs.dot(self.theta_omega, sigma[:, k] * y_range)

            # Δgrowth 항(간단 근사)
            T, H, L = x[0, k], x[1, k], x[3, k]
            gT = cs.exp(-((T - 25.0) ** 2) / (2 * 3 ** 2))
            gH = cs.exp(-((H - 70.0) ** 2) / (2 * 10 ** 2))
            gL = cs.exp(-((L - 300.0) ** 2) / (2 * 100.0 ** 2))
            growth = gT * gH * gL
            if k == 0:
                delta_growth = growth
            else:
                T_prev, H_prev, L_prev = x[0, k - 1], x[1, k - 1], x[3, k - 1]
                g_prev = cs.exp(-((T_prev - 25.0) ** 2) / (2 * 3 ** 2)) * \
                         cs.exp(-((H_prev - 70.0) ** 2) / (2 * 10 ** 2)) * \
                         cs.exp(-((L_prev - 300.0) ** 2) / (2 * 100.0 ** 2))
                delta_growth = growth - g_prev

            J += J_track + J_du + J_energy + J_slack - self.alpha_growth * delta_growth

        # (22h) Slack variable ≥ 0
        for k in range(N):
            g.append(sigma[:, k])
            lbg += [0.0] * nx
            ubg += [cs.inf] * nx

        # Terminal cost: θ_{y1,f} (y1(N) - r1)^2 (온도 터미널)
        J += self.theta_y1f * cs.sumsqr(x[0, N] - r[0])

        # NLP 구성
        w = cs.vertcat(
            cs.reshape(x, -1, 1),
            cs.reshape(u, -1, 1),
            cs.reshape(du, -1, 1),
            cs.reshape(sigma, -1, 1),
        )
        g = cs.vertcat(*g)
        p = cs.vertcat(x0, d, u_prev, r)

        prob = {"x": w, "f": J, "g": g, "p": p}
        # ✅ ipopt 내부 옵션으로 지정 (max_iter 오류 해결)
        opts = {
            "ipopt": {
                "max_iter": 50,
                "tol": 1e-4,
                "print_level": 0,
                "sb": "yes",
            },
            "print_time": False,
            "verbose": False,
        }
        self.solver = cs.nlpsol("solver", "ipopt", prob, opts)

        # 초기 guess & 제약 경계 저장
        self.w0 = np.zeros(((nx * (N + 1)) + (nu * N * 2) + (nx * N), 1))
        self.lbg = np.array(lbg, dtype=float).reshape(-1, 1)
        self.ubg = np.array(ubg, dtype=float).reshape(-1, 1)

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
                self.alpha_growth = float(theta["alpha_growth"])
            print(f"✅ Loaded RL parameters from {path}")
        except Exception as e:
            print(f"⚠️ Failed to load RL parameters: {e}")

    # ─────────────────────────────────────────────
    # 참조(목표값) 설정
    # ─────────────────────────────────────────────
    def set_reference(self, Tmid, Hmid, CO2_ref=420.0, L_ref=300.0):
        self.r = np.array([Tmid, Hmid, CO2_ref, L_ref], dtype=float)
        self.T_ref, self.H_ref = Tmid, Hmid

    # ─────────────────────────────────────────────
    # MPC policy: 제어 입력만 반환 (reward는 RL 계산)
    # ─────────────────────────────────────────────
    def policy(self, s: np.ndarray):
        # s = [state(4) , disturbance(4)]
        x_now = np.array(s[:4], dtype=float)
        d_now = np.array(s[4:8], dtype=float)
        p = np.concatenate([x_now, d_now, self.u_prev, self.r]).reshape(-1, 1)

        try:
            sol = self.solver(
                x0=self.w0,
                p=p,
                lbg=self.lbg,
                ubg=self.ubg,
                lbx=-1e9,
                ubx=+1e9,
            )
            w_opt = np.array(sol["x"]).flatten()
        except Exception as e:
            print(f"[WARN] CasADi solver failed: {e}")
            w_opt = self.w0.flatten()

        nx, nu, N = self._nx, self._nu, self.N
        # 첫 제어입력 u(0) 추출
        u0_start = (N + 1) * nx
        u_opt = w_opt[u0_start:u0_start + nu]
        u_opt = np.clip(u_opt, self.u_min, self.u_max)

        # 상태 업데이트용 warm-start 저장
        self.u_prev = u_opt.copy()
        self.w0 = w_opt.reshape(-1, 1)

        print(f"⚙️ u={u_opt.round(3)}")
        return u_opt
