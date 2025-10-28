# q_learning_greenhouse_pi.py
import time, signal, numpy as np
from learning_real_rl import LearningMpcReal
import os, csv

# ==== 하이퍼파라미터 (라즈베리파이 안전값) ====
TS = 60.0 * 15.0        # 15분 주기 (테스트시 더 짧게)
N  = 12                 # 3시간 horizon (경량화)
GAMMA = 0.99
LR = 1e-3               # SGD 학습률
UPDATE_EVERY = 8        # 8 step마다 θ 업데이트
MAX_DTHETA_FRAC = 0.05  # 5%/update 안전 클램프

def clamp_update(theta, dtheta):
    lim = np.maximum(np.abs(theta) * MAX_DTHETA_FRAC, 1e-3)
    return np.clip(dtheta, -lim, lim)

def main():
    print("🚀 RL-MPC Learning Loop Starting (Raspberry Pi)...")
    mpc = LearningMpcReal(prediction_horizon=N, constrain_control_rate=True)
    theta_keys = mpc.get_theta_keys()
    theta_path = "./theta_pi.json"

    # ✅ 이전 학습 파라미터 불러오기
    if os.path.exists(theta_path):
        mpc.load_theta_json(theta_path)
        theta = mpc.get_theta_vector(theta_keys)
        print(f"📁 Loaded existing theta parameters from {theta_path}")
    else:
        theta = mpc.get_theta_vector(theta_keys)
        print("🆕 Initialized new theta parameters")

    # ✅ 로그 파일 초기화
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "rl_mpc_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "timestamp", "temp", "hum", "co2", "light",
                "u_fan", "u_heater", "u_led",
                "reward", "TD_error",
                "theta_norm", "solve_status"
            ])
    print(f"🧾 Logging to {log_file}")

    last_u = np.zeros((3,), dtype=np.float32)
    step = 0
    stop = False
    signal.signal(signal.SIGINT, lambda *_: globals().update(stop=True))
    print("✅ MPC initialized.\n")

    while not stop:
        try:
            # 1️⃣ 현재 센서 상태
            x, y, y_prev, d = mpc.read_current_measurements()
            y_scalar = float(y[0])

            # 2️⃣ MPC 제어 입력 계산
            u, status = mpc.control_step(x, d)
            if status != "Solve_Succeeded":
                print("⚠️ Solver failed → 이전 제어값 사용")
                u = last_u
            mpc.apply_actuators(u)
            print(f"[STEP {step}] u_opt={u.round(3)}")

            # 3️⃣ 다음 상태 관측
            x2, y2, y_prev2, d2 = mpc.read_next_measurements()

            # 4️⃣ TD 학습 (저빈도)
            if step % UPDATE_EVERY == 0:
                q_sa = mpc.eval_Q(x, d, u)
                v_s2 = mpc.eval_V(x2, d2)
                r = mpc.compute_stage_reward(y_scalar, u)
                td = (r + GAMMA * (-v_s2)) - (-q_sa)
                grad = np.ones_like(theta, dtype=np.float32) * td
                dtheta = -LR * grad
                dtheta = clamp_update(theta, dtheta)
                theta = (theta + dtheta).astype(np.float32)
                mpc.set_theta_vector(theta, theta_keys)
                mpc.save_theta_json(theta_path, theta_keys, theta)
                print(f"🧠 TD={td:.6f}, θ updated & saved")

            last_u = u
            step += 1
            time.sleep(TS)

        except KeyboardInterrupt:
            print("\n🛑 중단 신호 감지 → 파라미터 저장 후 종료")
            mpc.save_theta_json(theta_path, theta_keys, theta)
            break
        except Exception as e:
            print(f"❌ Error at step {step}: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
