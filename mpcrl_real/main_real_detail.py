# main_real_detail_v2.py
# ──────────────────────────────────────────────
# Real-World RL-MPC (CasADi 기반, 논문 수식 (18)~(21) 완전 대응)
#   - 센서 → MPC → 액추에이터 → 보상 계산 루프
#   - 사람이 보기 쉬운 콘솔 로그 출력
# ──────────────────────────────────────────────

import time
import numpy as np
from learning_real_detail import LearningMpcCasADi   # ✅ CasADi MPC
from real_env import RealEnvironment
import sys
import os

# ────────────────────────────────
# 🧾 로그 폴더 자동 생성 + 로그 파일 열기
# ────────────────────────────────
os.makedirs("logs", exist_ok=True)
log_path = f"logs/rl_mpc_{time.strftime('%Y%m%d_%H%M%S')}.log"

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
print(f"📝 Logging to {log_path}")

# ──────────────────────────────────────────────
# RL-MPC 메인 루프 (논문식 보상 적용)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Real-world RL-MPC (CasADi) starting...")

    # 1️⃣ 환경 초기화
    env = RealEnvironment(sample_time=5.0)
    mpc = LearningMpcCasADi(ts=env.sample_time, N=24)

    # 2️⃣ 작물 프로필 기반 목표값 설정
    Tmid = sum(env.crop.get("target_temp", [18.0, 22.0])) * 0.5
    Hmid = sum(env.crop.get("target_humidity", [50.0, 70.0])) * 0.5
    mpc.set_reference(Tmid=Tmid, Hmid=Hmid, CO2_ref=420.0, L_ref=300.0)

    # 3️⃣ RL 버퍼 및 주기
    replay_buffer = []
    UPDATE_PERIOD = 3600 * 6
    last_update = time.time()

    step = 0
    u_prev = np.zeros(3)
    print("✅ RL-MPC loop running...\n")

    # ──────────────────────────────────────────────
    # 제어 루프 (논문 수식 기반 보상 적용)
    # ──────────────────────────────────────────────
    while True:
        step += 1
        t_now = time.time()

        # (a) 센서 & 외란 읽기
        x, d = env.read_sensors()
        s = np.concatenate([x, d])

        # (b) MPC 제어 계산
        u_opt = mpc.policy(s)
        env.send_actuators(u_opt)

        # (c) 논문식 보상 계산
        T_ref = mpc.T_ref
        H_ref = mpc.H_ref
        temp, hum, co2, light = x
        fan, heater, led = [float(np.clip(u, 0, 1)) for u in u_opt]

        du = u_opt - u_prev
        u_prev = u_opt.copy()

        # --- 가중치 (논문 대응) ---
        Q_T, Q_H = 1.0, 1.0
        R_dU = 0.05
        S_slack = 3.0
        c_energy = 0.02
        c_growth = 1.0

        # (1) 오차 항 (Tracking)
        err_T = (temp - T_ref)**2
        err_H = (hum - H_ref)**2
        J_track = Q_T * err_T + Q_H * err_H

        # (2) 제어 변화율
        J_delta = R_dU * np.sum(du**2)

        # (3) 제약 위반 항 (Slack)
        Tmin, Tmax = mpc.T_ref - 2.0, mpc.T_ref + 2.0
        Hmin, Hmax = mpc.H_ref - 10.0, mpc.H_ref + 10.0
        vT = max(0, Tmin - temp) + max(0, temp - Tmax)
        vH = max(0, Hmin - hum) + max(0, hum - Hmax)
        J_slack = S_slack * (vT**2 + vH**2)

        # (4) 에너지 비용
        J_energy = c_energy * (fan**2 + heater**2 + 0.5 * led**2)

        # (5) 성장 기여항
        G_T = np.exp(-0.5 * ((temp - 25.0)/2.5)**2)
        G_H = np.exp(-0.5 * ((hum - 60.0)/8.0)**2)
        G_L = np.tanh(light / 500.0)
        growth = G_T * G_H * G_L

        # (6) 최종 보상 (논문 수식 (18)~(21))
        reward = - (J_track + J_delta + J_slack + J_energy) + c_growth * growth

        print(f"🏆 Reward={reward:.4f} | Track={J_track:.3f} Δu={J_delta:.3f} Slack={J_slack:.3f} Energy={J_energy:.3f} Growth={growth:.3f}")

        # (d) 로그 출력
        print("\n────────────────────────────────────────────")
        print(f"⏱ [STEP {step:03d}] t={t_now:.2f} (Δt={env.sample_time:.1f}s)")
        print(f"🌡  Temp={x[0]:.1f}°C  💧Hum={x[1]:.1f}%  ☁️CO₂={x[2]:.0f}ppm  💡Light={x[3]:.1f}lx")
        print(f"🌞  Rad={d[0]:.0f}W/m²   🌬OutT={d[2]:.1f}°C  💧OutH={d[3]:.0f}%")
        print(f"⚙️  u_opt → FAN={u_opt[0]:.2f} | HEATER={u_opt[1]:.2f} | LED={u_opt[2]:.2f}")
        print(f"🏆 Reward = {reward: .4f}")
        print("────────────────────────────────────────────")

        # (e) 다음 상태 & 경험 저장
        x_next, d_next = env.read_sensors()
        s_next = np.concatenate([x_next, d_next])
        replay_buffer.append((s, u_opt, reward, s_next))

        # (f) RL 파라미터 업데이트 주기 처리
        if time.time() - last_update > UPDATE_PERIOD:
            # mpc.update_theta(replay_buffer)  # 필요 시 RL 업데이트 적용
            replay_buffer.clear()
            last_update = time.time()

        # (g) 제어 주기 대기
        time.sleep(env.sample_time)
