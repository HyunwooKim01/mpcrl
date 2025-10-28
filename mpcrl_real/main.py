# main_real_mpc.py
"""
실제 환경용 RL-MPC 제어 메인 루프 (real_env.py 사용하지 않음)
- MQTT로 센서 수신
- MPC로 최적 제어 입력 계산
- MQTT로 제어 명령 송신
"""

import time, json, os, numpy as np
from sims.configs.default_real import DefaultReal
from learning_real_rl import LearningMpcReal       # RL 통합형 MPC
from greenhouse.model_real import Model
from mqtt_handler import get_latest_sensor, publish_actuator


# ────────────────────────────────────────────────
# 🥬 작물 프로필 로드 (선택)
# ────────────────────────────────────────────────
def load_crop_profile(crop_name="lettuce"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "crops", f"{crop_name.lower()}_profile.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            profile = json.load(f)
        print(f"🌿 Loaded crop profile: {profile.get('crop','unknown')} @ {path}")
        return profile
    else:
        print(f"⚠️ {path} not found → DefaultReal 사용")
        return None


# ────────────────────────────────────────────────
# 🚀 메인 제어 루프
# ────────────────────────────────────────────────
def main():
    print("🚀 RL-MPC Real Control Loop Starting...")

    crop_name = "lettuce"
    crop_profile = load_crop_profile(crop_name)
    crop_config = DefaultReal(crop_name)
    mpc = LearningMpcReal(test=crop_config)
    print("✅ MPC initialized.\n")

    CONTROL_PERIOD = 15 * 60   # 15분 주기 (센서 주기 맞춤)

    while True:
        try:
            # 1️⃣ MQTT에서 최신 센서데이터 수신
            sensor_data = get_latest_sensor()
            if not sensor_data:
                print("⏳ 대기중: 센서데이터 수신 안됨 (MQTT)")
                time.sleep(5)
                continue

            # 2️⃣ 센서데이터 파싱
            temp = float(sensor_data.get("temp", 0))
            hum = float(sensor_data.get("hum", 0))
            co2 = float(sensor_data.get("co2", 0))
            light = float(sensor_data.get("light", 0))

            # 상태(x), 외란(d) 구성
            x_current = np.array([0.0, co2, temp, hum])   # 내부상태
            d_current = np.array([light, co2, temp, hum]) # 외란

            print(f"\n📡 [Sensor] Temp={temp:.1f}°C  Hum={hum:.1f}%  CO₂={co2:.0f}ppm  Light={light:.0f}lx")

            # 3️⃣ MPC 계산
            u_opt, status = mpc.compute_control(x_current, d_current)
            print(f"[MPC] status={status}, u_opt={u_opt.round(3)}")

            # 4️⃣ MQTT로 제어값 송신
            publish_actuator(u_opt)

            # 5️⃣ 주기 대기
            time.sleep(CONTROL_PERIOD)

        except KeyboardInterrupt:
            print("\n🛑 제어 루프 중단 (사용자)")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
