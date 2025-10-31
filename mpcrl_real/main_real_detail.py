# main_real_detail.py
# ──────────────────────────────────────────────
# Real-World MPC (CasADi 기반, RL 분리 버전)
#   - 센서 → MPC → 액추에이터 제어 루프
#   - 보상 및 파라미터 업데이트는 외부 RL 모듈(Q_learning.py)에서 수행
# ──────────────────────────────────────────────

import time
import numpy as np
from learning_real_detail import LearningMpcCasADi   # ✅ CasADi MPC
from real_env import RealEnvironment
import sys
import os

# ────────────────────────────────
# 🧾 로그 폴더 자동 생성 + 파일 출력
# ────────────────────────────────
os.makedirs("logs", exist_ok=True)
log_path = f"logs/mpc_{time.strftime('%Y%m%d_%H%M%S')}.log"

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
# Main Loop
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Real-world MPC (CasADi) starting...")

    # 1️⃣ 환경 초기화
    env = RealEnvironment(sample_time=5.0)
    mpc = LearningMpcCasADi(ts=env.sample_time, N=24)

    # 2️⃣ 작물 프로필 기반 목표값 설정
    Tmid = sum(env.crop.get("target_temp", [18.0, 22.0])) * 0.5
    Hmid = sum(env.crop.get("target_humidity", [50.0, 70.0])) * 0.5
    mpc.set_reference(Tmid=Tmid, Hmid=Hmid, CO2_ref=420.0, L_ref=300.0)
    print(f"🎯 Target → T_ref={mpc.T_ref:.1f}°C, H_ref={mpc.H_ref:.1f}%")

    # 3️⃣ 루프 설정
    replay_buffer = []         # Q-learning에서 사용 가능
    UPDATE_PERIOD = 3600 * 6   # RL 업데이트 주기(선택)
    last_update = time.time()
    step = 0

    print("✅ MPC loop running...\n")

    # ──────────────────────────────────────────────
    # 제어 루프 (RL reward 계산 제거)
    # ──────────────────────────────────────────────
    while True:
        step += 1
        t_now = time.time()

        # (a) 센서 & 외란 읽기
        x, d = env.read_sensors()
        s = np.concatenate([x, d])

        # (b) MPC 제어 계산
        u_opt = mpc.policy(s)

        # (c) 액추에이터 명령 전송
        env.send_actuators(u_opt)

        # (d) 상태 로깅
        print("\n────────────────────────────────────────────")
        print(f"⏱ [STEP {step:03d}] t={t_now:.2f} (Δt={env.sample_time:.1f}s)")
        print(f"🌡 Temp={x[0]:.1f}°C  💧Hum={x[1]:.1f}%  ☁️CO₂={x[2]:.0f}ppm  💡Light={x[3]:.1f}lx")
        print(f"🌞 Rad={d[0]:.0f}W/m²  🌬OutT={d[2]:.1f}°C  💧OutH={d[3]:.0f}%")
        print(f"⚙️ u_opt → FAN={u_opt[0]:.2f} | HEATER={u_opt[1]:.2f} | LED={u_opt[2]:.2f}")
        print("────────────────────────────────────────────")

        # (e) 다음 상태 저장 (RL 모듈용)
        x_next, d_next = env.read_sensors()
        s_next = np.concatenate([x_next, d_next])
        replay_buffer.append((s, u_opt, s_next))  # reward 제외

        # (f) RL 업데이트 트리거 (선택)
        if time.time() - last_update > UPDATE_PERIOD:
            # 외부 Q_learning.py가 θ 업데이트 담당
            replay_buffer.clear()
            last_update = time.time()

        # (g) 제어 주기 대기
        time.sleep(env.sample_time)
