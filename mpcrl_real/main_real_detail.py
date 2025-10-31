# main_real_detail_clean.py
# ──────────────────────────────────────────────
# Real-World RL-MPC (CasADi 기반, 논문 수식 (18)~(21))
#   - RL 파라미터 로드 & 주기적 업데이트
#   - 보상 계산은 learning_real_detail 내부에서 수행
# ──────────────────────────────────────────────

import time
import numpy as np
from learning_real_detail import LearningMpcCasADi   # ✅ CasADi MPC
from real_env import RealEnvironment
import sys
import os
import pickle

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
# RL-MPC 메인 루프
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Real-world RL-MPC (CasADi) starting...")

    # 1️⃣ 환경 초기화
    env = RealEnvironment(sample_time=5.0)
    mpc = LearningMpcCasADi(ts=env.sample_time, N=24)

    # 2️⃣ RL 학습된 파라미터 적용 (trained_theta.pkl)
    try:
        mpc.load_theta("trained_theta.pkl")   # ✅ RL 학습 결과 적용
    except Exception as e:
        print(f"⚠️ RL 파라미터 로드 실패: {e}")

    # 3️⃣ 작물 프로필 기반 목표값 설정
    Tmid = sum(env.crop.get("target_temp", [18.0, 22.0])) * 0.5
    Hmid = sum(env.crop.get("target_humidity", [50.0, 70.0])) * 0.5
    mpc.set_reference(Tmid=Tmid, Hmid=Hmid, CO2_ref=420.0, L_ref=300.0)
    print(f"🎯 Set references → T_ref={mpc.T_ref:.1f}°C, H_ref={mpc.H_ref:.1f}%")

    # 4️⃣ RL 버퍼 및 업데이트 주기 설정
    replay_buffer = []
    UPDATE_PERIOD = 3600 * 6   # 6시간마다 업데이트
    last_update = time.time()

    step = 0
    print("✅ RL-MPC loop running...\n")

    # ──────────────────────────────────────────────
    # 제어 루프
    # ──────────────────────────────────────────────
    while True:
        step += 1
        t_now = time.time()

        # (a) 센서 & 외란 읽기
        x, d = env.read_sensors()
        s = np.concatenate([x, d])

        # (b) MPC 제어 + 내부 보상 계산
        u_opt, reward = mpc.policy(s)
        env.send_actuators(u_opt)

        # (c) 다음 상태 & 경험 저장
        x_next, d_next = env.read_sensors()
        s_next = np.concatenate([x_next, d_next])
        replay_buffer.append((s, u_opt, reward, s_next))

        # (d) RL 파라미터 업데이트 주기 처리
        if time.time() - last_update > UPDATE_PERIOD:
            print("\n🧠 Updating RL parameters (θ = {Q,R,S,α}) with "
                  f"{len(replay_buffer)} samples...\n")
            mpc.update_theta(replay_buffer)

            # 학습된 결과를 다시 저장
            new_theta = {
                "Q": np.diag(mpc.Q).tolist(),
                "R": np.diag(mpc.R).tolist(),
                "S": np.diag(mpc.S).tolist(),
                "alpha_growth": mpc.alpha_growth
            }
            with open("trained_theta.pkl", "wb") as f:
                pickle.dump(new_theta, f)
            print("💾 Updated θ saved to trained_theta.pkl")

            print(f"🔧 Current θ → Q={np.diag(mpc.Q)}, R={np.diag(mpc.R)}, "
                  f"S={np.diag(mpc.S)}, α={mpc.alpha_growth:.3f}\n")

            replay_buffer.clear()
            last_update = time.time()

        # (e) 제어 주기 대기
        time.sleep(env.sample_time)
