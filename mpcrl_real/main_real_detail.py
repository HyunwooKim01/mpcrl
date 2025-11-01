# main_real_detail.py
# ──────────────────────────────────────────────
# Real-World MPC Controller (CasADi 기반)
#   - 센서 → MPC → 액추에이터 제어 루프
#   - RL(Q-learning_mqtt.py)은 학습만 수행
#   - MPC는 farmA/esp1/mpc/u_opt 토픽으로 제어값 publish
# ──────────────────────────────────────────────

import time, os, sys, json
import numpy as np
import paho.mqtt.client as mqtt     # ✅ RL Learner 통신용
from learning_real_detail import LearningMpcCasADi
from real_env import RealEnvironment

# ────────────────────────────────
# 🧾 로그 파일 설정
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

# ────────────────────────────────
# 🚀 Main MPC Loop
# ────────────────────────────────
if __name__ == "__main__":
    print("🚀 Real-world MPC (CasADi) starting...")

    # 1️⃣ 환경 초기화
    farm_id = "farmA"
    esp_id = "esp1"
    broker_host = "211.106.231.24"
    broker_port = 1883
    env = RealEnvironment(sample_time=5.0,
                          broker_host=broker_host,
                          broker_port=broker_port,
                          farm_id=farm_id,
                          esp_id=esp_id)

    # 2️⃣ MPC 초기화
    try:
        mpc = LearningMpcCasADi(ts=env.sample_time, N=24)
    except TypeError:
        mpc = LearningMpcCasADi()

    # 3️⃣ 목표값 설정 (lettuce_profile.json 기반)
    Tmid = sum(env.crop.get("target_temp", [18.0, 22.0])) * 0.5
    Hmid = sum(env.crop.get("target_humidity", [50.0, 70.0])) * 0.5
    try:
        mpc.set_reference(Tmid=Tmid, Hmid=Hmid, CO2_ref=420.0, L_ref=300.0)
    except Exception:
        pass
    print(f"🎯 Target → T_ref={Tmid:.1f}°C, H_ref={Hmid:.1f}%")

    # 4️⃣ MQTT (RL learner에게 u_opt 전달)
    def on_disconnect(client, userdata, rc):
        print("⚠️ MQTT disconnected! Retrying...")
        while True:
            try:
                client.reconnect()
                print("🔁 MQTT reconnected successfully!")
                break
            except Exception as e:
                print(f"❌ Reconnect failed: {e}")
                time.sleep(5)
                
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_disconnect = on_disconnect
    
    # ✅ 연결 시도 (성공할 때까지 반복)
    while True:
        try:
            mqtt_client.connect(broker_host, broker_port, keepalive=30)
            print(f"✅ MQTT connected → {broker_host}:{broker_port}")
            break
        except Exception as e:
            print(f"❌ MQTT connection failed: {e}")
            print("🔁 5초 후 다시 시도합니다...")
            time.sleep(5)

    mqtt_client.loop_start()
    topic_uopt = f"{farm_id}/{esp_id}/mpc/u_opt"
    print(f"📡 RL learner publish topic: {topic_uopt}")

    # 5️⃣ 루프 실행
    step = 0
    print("✅ MPC loop running...\n")

    try:
        while True:
            step += 1
            t_now = time.time()

            # (a) 센서 & 외란 읽기
            x, d = env.read_sensors()
            s = np.concatenate([x, d])

            # (b) MPC 제어 계산
            try:
                out = mpc.policy(s)
                if isinstance(out, tuple):
                    u_opt = np.array(out[0], dtype=float).reshape(-1)
                else:
                    u_opt = np.array(out, dtype=float).reshape(-1)
            except Exception as e:
                print(f"[WARN] MPC policy failed: {e}")
                u_opt = np.zeros(3, dtype=float)

            # (c) 액추에이터 명령 전송
            env.send_actuators(u_opt)

            # (d) RL Learner로 u_opt publish
            payload = {
                "fan": float(u_opt[0]),
                "heater": float(u_opt[1]),
                "led": float(u_opt[2]),
                "timestamp": time.time()
            }
            mqtt_client.publish(topic_uopt, json.dumps(payload), qos=0)
            print(f"📡 Published MPC u_opt → {payload}")

            # (e) 상태 로그 출력
            print("\n────────────────────────────────────────────")
            print(f"⏱ [STEP {step:03d}] t={t_now:.2f} (Δt={env.sample_time:.1f}s)")
            print(f"🌡 Temp={x[0]:.1f}°C  💧Hum={x[1]:.1f}%  ☁️CO₂={x[2]:.0f}ppm  💡Light={x[3]:.1f}lx")
            print(f"🌞 Rad={d[0]:.0f}W/m²  🌬OutT={d[2]:.1f}°C  💧OutH={d[3]:.0f}%")
            print(f"⚙️ u_opt → FAN={u_opt[0]:.2f} | HEATER={u_opt[1]:.2f} | LED={u_opt[2]:.2f}")
            print("────────────────────────────────────────────")

            # (f) 주기 대기
            time.sleep(env.sample_time)

    except KeyboardInterrupt:
        print("\n🛑 Stopping MPC loop...")
        mqtt_client.loop_stop()
        log_file.close()
        print("✅ Clean exit.")
