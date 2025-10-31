# learning_real_casadi_fixed_v3.py
# ✅ CasADi 기반 MPC + RL 파라미터 로드 기능 추가 (논문식 (18)~(21) 대응)
from __future__ import annotations
import time
import numpy as np
import casadi as cs
import pickle


class LearningMpcCasADi:
    """CasADi 기반 MPC (Raspberry Pi 실시간용)
    상태 x=[temp_in,hum_in,co2_in,light_in]
    제어 u=[fan,heater,led]
    외란 d=[solar_rad,co2_out,temp_out,hum_out]
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
        # reward_log 제거
        # last_update 제거 가능

        # MPC 파라미터 초기값
        self.theta_dy1 = 100.0
        self.theta_u_vec = np.array([10.0, 1.0, 1.0])
        self.theta_w_vec = np.array([1e5, 1e5])
        self.Q = np.diag([2.0, 1.0, 0.2, 0.5])
        self.R = np.diag([0.05, 0.1, 0.1])
        self.S = np.diag([0.2, 0.3, 0.3])
        self.alpha_growth = 1.0
        self._build_mpc()

        nx, nu, nd = 4, 3, 4
        self._nx, self._nu = nx, nu

        # --- Symbolic Variables ---
        x = cs.MX.sym("x", nx, N + 1)
        u = cs.MX.sym("u", nu, N)
        du = cs.MX.sym("du", nu, N)
        x0 = cs.MX.sym("x0", nx)
        d = cs.MX.sym("d", nd)
        u_prev = cs.MX.sym("u_prev", nu)
        r = cs.MX.sym("r", nx)

        # --- Dynamics ---
        def f_dyn(xk, uk, dk):
            temp, hum, co2, light = xk[0], xk[1], xk[2], xk[3]
            fan, heater, led = uk[0], uk[1], uk[2]
            rad, co2_out, temp_out, hum_out = dk[0], dk[1], dk[2], dk[3]

            dtemp = 0.015 * (temp_out - temp) + 0.1 * heater - 0.07 * fan
            dhum = 0.01 * (hum_out - hum) - 0.06 * fan + 0.015 * heater
            dco2 = 0.002 * (co2_out - co2)
            dlight = -0.05 * light + 1.2 * led * 300.0 + 0.15 * rad
            return cs.vertcat(dtemp, dhum, dco2, dlight)

        # --- Constraints / Objective ---
        g = []
        J = 0
        g.append(x[:, 0] - x0)

        for k in range(N):
            x_next = x[:, k] + ts * f_dyn(x[:, k], u[:, k], d)
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
            Tmin, Tmax = 18.0, 28.0
            Hmin, Hmax = 40.0, 80.0
            vT = cs.fmax(0, Tmin - T) + cs.fmax(0, T - Tmax)
            vH = cs.fmax(0, Hmin - H) + cs.fmax(0, H - Hmax)
            S_slack = 3.0
            J_slack = S_slack * (vT * vT + vH * vH)

            gT = cs.exp(-0.5 * cs.power((T - 25.0) / 2.5, 2))
            gH = cs.exp(-0.5 * cs.power((H - 60.0) / 8.0, 2))
            gL = cs.tanh(x[3, k] / 500.0)
            growth = gT * gH * gL

            J += J_track + J_du + J_energy + J_slack - self.alpha_growth * growth

        J += 0.5 * cs.mtimes([(x[:, N] - r).T, self.Q, (x[:, N] - r)])

        w = cs.vertcat(cs.reshape(x, -1, 1), cs.reshape(u, -1, 1), cs.reshape(du, -1, 1))
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

        self.w0 = np.zeros(((nx * (N + 1)) + (nu * N * 2), 1))

    # -------------------- RL 파라미터 로드 기능 --------------------
    def load_theta(self, path: str):
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

    # --------------------
    def set_reference(self, Tmid, Hmid, CO2_ref=420.0, L_ref=300.0):
        self.r = np.array([Tmid, Hmid, CO2_ref, L_ref])
        self.T_ref, self.H_ref = Tmid, Hmid

    # --------------------
    def policy(self, s: np.ndarray):
        """MPC policy: 입력 u_opt만 계산 (reward는 RL이 계산)"""
        x = np.array(s[:4])
        d = np.array(s[4:8])
        p = np.concatenate([x, d, self.u_prev, self.r]).reshape(-1,1)
        nx, nu, N = self._nx, self._nu, self.N
        n_eq = (N+1)*nx + N*nu

        try:
            sol = self.solver(x0=self.w0, p=p,
                            lbg=np.zeros((n_eq,1)),
                            ubg=np.zeros((n_eq,1)),
                            lbx=-1e9, ubx=+1e9)
        except Exception as e:
            print(f"[WARN] CasADi solver failed: {e}")
            sol = {"x": np.zeros_like(self.w0)}

        w_opt = np.array(sol["x"]).flatten()
        u_opt = w_opt[(N+1)*nx:(N+1)*nx+nu]
        u_opt = np.clip(u_opt, 0.0, 1.0)
        self.u_prev = u_opt.copy()
        self.w0 = w_opt.reshape(-1,1)

        print(f"⚙️ u={u_opt.round(3)}")
        return u_opt
