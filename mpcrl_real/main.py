"""
main_real.py
-------------
실제환경용 MPC 제어 루프
센서 데이터 → MPC 계산 → 액추에이터 제어 순으로 반복 실행
"""

from learning_real import LearningMpcReal
from sims.configs.default_real import DefaultReal
from real_env import RealEnvironment
import numpy as np

if __name__ == "__main__":
    print("🚀 Real-world MPC controller starting...")

    # 1️⃣ 실제 환경 객체 초기화
    # env = RealEnvironment(sample_time=60.0 * 15.0)  # 15분 주기
    env = RealEnvironment(sample_time=5.0)  # 테스트용. 5초마다 제어 루프 실행
    
    # 2️⃣ MPC 제어기 초기화
    test = DefaultReal()
    mpc = LearningMpcReal(test=test)
    print("✅ MPC controller initialized.")

    # 3️⃣ 제어 루프
    while True:
        try:
            # (1) 센서 상태 & 외란 읽기
            x_current = env.read_sensors()
            d_current = env.read_disturbance()

            print(f"[SENSOR] x={x_current.round(3)}, d={d_current.round(3)}")

            # (2) MPC 계산
            u_opt, status = mpc.compute_control(x_current, d_current)
            print(f"[MPC] status={status}, u_opt={u_opt.round(3)}")

            # (3) 액추에이터 적용
            env.apply_control(u_opt)

            # (4) 다음 주기 대기
            env.wait_next_cycle()

        except KeyboardInterrupt:
            print("🛑 MPC control loop stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break
