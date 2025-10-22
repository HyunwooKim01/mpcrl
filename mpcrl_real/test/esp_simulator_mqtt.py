"""
esp_simulator_mqtt.py
---------------------
MQTT를 통해 하루 주기의 스마트팜 센서/외란 데이터를 송출하는 시뮬레이터.
MPC 테스트용으로 사용.
- 주/야 주기 및 교란 이벤트 포함
- 1초마다 데이터 전송 (speed_factor 배속으로 시간 가속)
"""

import time
import math
import json
import paho.mqtt.client as mqtt

print("🚀 MQTT 연결 시도 중...")

try:
    client = mqtt.Client()
    client.connect("211.106.231.24", 1883, 60)
    print("✅ 연결 성공")
except Exception as e:
    print("❌ 연결 실패:", e)


class EspSimulatorMQTT:
    def __init__(
        self,
        broker_ip="211.106.231.24",
        farm_id="farmA",
        esp_id="esp1",
        speed_factor=60.0,
    ):
        """
        broker_ip: MQTT 브로커 IP
        farm_id  : 농장 ID
        esp_id   : ESP ID
        speed_factor: 시간 가속 배율 (60 = 1분이 1시간)
        """
        self.client = mqtt.Client()
        self.client.connect(broker_ip, 1883, 60)
        self.client.loop_start()
        self.farm_id = farm_id
        self.esp_id = esp_id
        self.speed_factor = speed_factor
        self.t = 0.0  # 누적 시간(분 단위)
        print(f"🚀 MQTT ESP Simulator started ({broker_ip})")

    def _publish(self, topic, payload):
        msg = json.dumps(payload)
        self.client.publish(topic, msg)
        print(f"[MQTT→Pi] {topic} : {msg}")

    def _simulate_day_cycle(self):
        """24시간 주기 + 교란 이벤트 포함 환경 생성"""
        hour = (self.t / 60.0) % 24  # 현재 시각 (0~24h)

        # ☀️ 복사량: 6~18시 사이만 존재 (낮)
        rad = max(0, math.sin((hour - 6) / 12 * math.pi))  # 0~1

        # 🌡 외기온도 / 습도 변화 (일주기)
        out_temp = 18.0 + 7.0 * rad  # 18~25°C
        out_hum = 80.0 - 25.0 * rad  # 80→55%

        # 🌿 내부 환경 기본값
        temp = 23.0 + 6.0 * rad + (0.5 * math.sin(self.t * 0.05))  # 22~29°C
        hum = 60.0 + 15.0 * math.cos(self.t * 0.04 + 1)             # 45~75%
        leaf = 0.007 + 0.001 * math.sin(self.t * 0.1)               # 0.006~0.008
        biomass = 0.003 + 0.001 * math.sin(self.t * 0.03)           # 0.002~0.004

        # 🌪️ 인위적 교란 시나리오 추가
        if 12 <= hour < 13:
            # 🔥 한낮 과열 이벤트
            temp += 5.0   # 온실 과열
            hum -= 10.0   # 건조
            event = "🔥 Overheat"
        elif 18 <= hour < 19:
            # ❄️ 저녁 냉각 이벤트
            temp -= 5.0   # 급냉
            hum += 15.0   # 결로 위험
            event = "❄️ Cooling"
        else:
            event = "☀️ Normal"

        # 외란 publish
        dist_payload = {
            "radiation": rad * 250.0,          # 0~250 W/m²
            "outside_temp": out_temp,          # 18~25°C
            "outside_humidity": out_hum,       # 55~80%
            "hour": hour
        }
        self._publish(f"{self.farm_id}/{self.esp_id}/disturbance", dist_payload)

        # 센서 publish
        sensor_payload = {
            "temperature": temp,
            "humidity": hum,
            "leaf_water": leaf,
            "biomass": biomass
        }
        self._publish(f"{self.farm_id}/{self.esp_id}/sensor", sensor_payload)

        # 🕒 콘솔 로그 출력
        print(
            f"🕒 {hour:05.2f}h | T={temp:5.2f}°C | H={hum:5.1f}% | "
            f"rad={rad*250:6.1f} | Event={event}"
        )

    def run(self):
        """시뮬레이터 루프"""
        try:
            while True:
                self._simulate_day_cycle()
                time.sleep(1)  # 1초마다 MQTT 전송
                self.t += self.speed_factor * 2  # 2분씩 시간 경과
        except KeyboardInterrupt:
            print("🛑 Simulation stopped.")
            self.client.loop_stop()


if __name__ == "__main__":
    try:
        sim = EspSimulatorMQTT(
            broker_ip="211.106.231.24",  # MQTT 브로커 IP
            farm_id="farmA",
            esp_id="esp1",
            speed_factor=60.0,           # 1초 = 1분
        )
        print("🚀 Simulator initialized. Starting loop...")
        sim.run()

    except Exception as e:
        import traceback
        print("❌ 프로그램 중단됨:")
        traceback.print_exc()
