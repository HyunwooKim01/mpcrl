# learning_real_detail.py
# ✅ CasADi 기반 MPC + RL 업데이트 + 보상/로깅 통합 (논문식 (18)~(21))
from __future__ import annotations
import time
import numpy as np
import casadi as cs
import pickle
import torch
import os

class LearningMpcCasADi:
    def __init__(self, ts: float = 5.0, N: int = 24, **kwargs):
        # 기본 설정
        self.ts = ts
        self.N = N
        self._nx, self._nu = 4, 3
        self.u_prev = np.zeros(3)
        self.r = np.array([20.0, 60.0, 420.0, 300.0])
        self.t0 = time.time()
        self.reward_log = []
        self.last_update = time.time()

        # RL-MPC 파라미터 초기값
        self.theta_dy1 = 100.0
        self.theta_u_vec = np.array([10.0, 1.0, 1.0])
        self.theta_w_vec = np.array([1e5, 1e5])
        self.gamma = 0.99
        self.Q = np.diag([2.0, 1.0, 0.2, 0.5])
        self.R = np.diag([0.05, 0.1, 0.1])
        self.S = np.diag([0.2, 0.3, 0.3])
        self.alpha_growth = 1.0

        # MPC 모델 구성
        self._build_mpc()

    # ─────────────────────────────
    # 🔧 MPC 모델 구축
    # ─────────────────────────────
    def _build_mpc(self):
        nx, nu, nd, N = self._nx, self._nu, 4, self.N
        x = cs.MX.sym("x", nx, N+1)
        u = cs.MX.sym("u", nu, N)
        du = cs.MX.sym("du", nu, N)
        x0 = cs.MX.sym("x0", nx)
        d = cs.MX.sym("d", nd)
        u_prev = cs.MX.sym("u_prev", nu)
        r = cs.MX.sym("r", nx)

        # ────── 단순화된 온실 동역학 ──────
        def f_dyn(xk, uk, dk):
            temp  = xk[0]
            hum   = xk[1]
            co2   = xk[2]
            light = xk[3]

            fan, heater, led = uk[0], uk[1], uk[2]

            temp_out = dk[0]
            hum_out  = dk[1]
            co2_out  = dk[2]
            solar_rad = dk[3]

            dtemp = -0.1 * fan + 0.1 * heater + 0.01 * (solar_rad - temp_out)
            dhum  = -0.05 * fan + 0.02 * heater + 0.01 * (hum_out - hum)
            dco2  = -0.01 * fan + 0.05 * co2_out - 0.02 * co2
            dlight = led * 0.1
            return cs.vertcat(dtemp, dhum, dco2, dlight)

        g = []
        J = 0
        g.append(x[:,0] - x0)

        for k in range(N):
            x_next = x[:,k] + self.ts * f_dyn(x[:,k], u[:,k], d)
            g.append(x[:,k+1] - x_next)
            if k == 0:
                g.append(du[:,k] - (u[:,k]-u_prev))
            else:
                g.append(du[:,k] - (u[:,k]-u[:,k-1]))

            # 성장 및 페널티 계산
            T, H, L = x[0,k], x[1,k], x[3,k]
            gT = cs.exp(-0.5 * ((T-25.0)/2.5)**2)
            gH = cs.exp(-0.5 * ((H-60.0)/8.0)**2)
            gL = cs.tanh(L/500.0)
            growth = gT*gH*gL
            L_y1 = -self.theta_dy1 * growth
            L_u = cs.dot(cs.DM(self.theta_u_vec), cs.fabs(u[:,k]))
            vT = cs.fmax(0,18.0-T)+cs.fmax(0,T-28.0)
            vH = cs.fmax(0,40.0-H)+cs.fmax(0,H-80.0)
            L_psi = cs.dot(cs.DM(self.theta_w_vec), cs.vertcat(vT,vH))
            L_du = cs.mtimes([du[:,k].T, self.R, du[:,k]])
            J += (self.gamma**k)*(L_y1+L_u+L_psi+L_du)

        J += 0.5*cs.mtimes([(x[:,N]-r).T, self.Q, (x[:,N]-r)])
        w = cs.vertcat(cs.reshape(x,-1,1),cs.reshape(u,-1,1),cs.reshape(du,-1,1))
        g = cs.vertcat(*g)
        p = cs.vertcat(x0,d,u_prev,r)
        nlp = {"x":w,"f":J,"g":g,"p":p}

        self.solver = cs.nlpsol("solver","sqpmethod",nlp,{
            "max_iter":40,
            "print_time":False,
            "print_header":False
        })
        self.w0 = np.zeros(((nx*(N+1))+(nu*N*2),1))

    # ─────────────────────────────
    # 🧠 RL 파라미터 로드
    # ─────────────────────────────
    def load_theta(self, path="trained_theta.pkl"):
        if not os.path.exists(path):
            print(f"⚠️ No trained_theta.pkl found at {path}, using default θ values")
            return
        try:
            with open(path, "rb") as f:
                theta = pickle.load(f)
            if "Q" in theta: self.Q = np.diag(theta["Q"])
            if "R" in theta: self.R = np.diag(theta["R"])
            if "S" in theta: self.S = np.diag(theta["S"])
            if "alpha_growth" in theta: self.alpha_growth = theta["alpha_growth"]
            print(f"✅ Loaded trained RL parameters from {path}")
        except Exception as e:
            print(f"⚠️ Error loading θ: {e}")

    # ─────────────────────────────
    # 🎯 Reference 설정
    # ─────────────────────────────
    def set_reference(self, Tmid=18.0, Hmid=65.0, CO2_ref=420.0, L_ref=300.0):
        """Set greenhouse target references."""
        self.r = np.array([Tmid, Hmid, CO2_ref, L_ref])
        self.T_ref = Tmid
        self.H_ref = Hmid
        print(f"🎯 Reference updated → T={Tmid}°C, H={Hmid}%, CO₂={CO2_ref}, L={L_ref}")

    # ─────────────────────────────
    # ⚙️ 정책 실행 + 보상 계산
    # ─────────────────────────────
    def policy(self, s: np.ndarray):
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

        reward = self._compute_reward(x, u_opt)
        self.reward_log.append(reward)
        print(f"⚙️ u={u_opt.round(3)} | 🏆 reward={reward:.4f}")
        return u_opt, reward

    # ─────────────────────────────
    # 🧮 보상 계산 (논문 수식 18~21 대응)
    # ─────────────────────────────
    def _compute_reward(self, x, u):
        T, H, CO2, L = x
        fan, heater, led = u
        T_ref, H_ref = self.r[0], self.r[1]
        err_T, err_H = (T-T_ref)**2, (H-H_ref)**2
        J_track = err_T + err_H
        du = u - self.u_prev
        J_delta = 0.05*np.sum(du**2)
        J_slack = 3.0*(max(0,T-28)**2 + max(0,18-T)**2)
        J_energy = 0.02*(fan**2+heater**2+0.5*led**2)
        G_T = np.exp(-0.5*((T-25)/2.5)**2)
        G_H = np.exp(-0.5*((H-60)/8)**2)
        G_L = np.tanh(L/500)
        growth = G_T*G_H*G_L
        reward = -(J_track+J_delta+J_slack+J_energy) + self.alpha_growth*growth
        return float(reward)

    # ─────────────────────────────
    # 🧩 RL 파라미터 업데이트 (Q-learning 스타일)
    # ─────────────────────────────
    def update_theta(self, replay_buffer, gamma=0.99, lr=0.1):
        if len(replay_buffer) < 5:
            print("⚠️ Not enough samples for RL update.")
            return

        Q = torch.tensor(np.diag(self.Q),dtype=torch.float32,requires_grad=True)
        R = torch.tensor(np.diag(self.R),dtype=torch.float32,requires_grad=True)
        S = torch.tensor(np.diag(self.S),dtype=torch.float32,requires_grad=True)
        alpha_growth = torch.tensor(self.alpha_growth,dtype=torch.float32,requires_grad=True)
        losses=[]
        for (s,u,r,s_next) in replay_buffer:
            s_t = torch.tensor(s,dtype=torch.float32)
            u_t = torch.tensor(u,dtype=torch.float32)
            r_t = torch.tensor(r,dtype=torch.float32)
            Q_est = (s_t[:4]-torch.tensor(self.r[:4])).pow(2).dot(Q)+u_t.pow(2).dot(R)+u_t.abs().dot(S)
            V_next = 0.5*Q_est.detach()
            td_error = r_t + gamma*V_next - Q_est
            losses.append(td_error**2)

        loss = torch.mean(torch.stack(losses))
        loss.backward()
        with torch.no_grad():
            Q -= lr*Q.grad; R -= lr*R.grad; S -= lr*S.grad
            alpha_growth -= lr*alpha_growth.grad
        self.Q = np.diag(Q.detach().numpy())
        self.R = np.diag(R.detach().numpy())
        self.S = np.diag(S.detach().numpy())
        self.alpha_growth = float(alpha_growth.detach().numpy())

        print("✅ θ updated (Q,R,S,α) via Q-learning.")
