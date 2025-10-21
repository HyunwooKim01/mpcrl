"""
esp_simulator_mqtt.py
---------------------
ESP 없이 MQTT를 통해 하루 주기의 스마트팜 센서/외란 데이터를 자동 송출하는 시뮬레이터.
MPC 테스트용으로 사용.
"""

import time
import math
import json
import paho.mqtt.client as mqtt


class EspSimulatorMQTT:
    def __init__(self,
                 broker_ip="localhost",
                 farm_id="farmA",
                 esp_id="esp1",
                 speed_factor=60.0):
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
        self.t = 0.0  # "시간 (분)" 기준 누적 시뮬레이션 시간

        print(f"🚀 MQTT ESP Simulator started ({broker_ip})")

    def _publish(self, topic, payload):
        msg = json.dumps(payload)
        self.client.publish(topic, msg)
        print(f"[MQTT→Pi] {topic} : {msg}")

    def _simulate_day_cycle(self):
        """하루(24시간) 주기로 외기와 내부 센서 패턴 변화 생성"""
        hour = (self.t / 60.0) % 24  # 0~24시간

        # ☀️ 복사량 (낮: 6~18시)
        rad = max(0, math.sin((hour - 6) / 12 * math.pi))  # 0~1
        # 🌡 외기온 (낮엔 높고 밤엔 낮음)
        out_temp = 20.0 + 5.0 * rad  # 20~25°C
        # 💧 외기습도 (낮엔 낮고 밤엔 높음)
        out_hum = 70.0 - 10.0 * rad  # 70→60%

        # 내부 상태 (단순 모델)
        temp = 24.0 + 3.0 * rad + (0.5 - math.sin(self.t * 0.05))  # 23~28°C
        hum = 60.0 + 10.0 * math.cos(self.t * 0.05 + 1)
        leaf = 0.007 + 0.0008 * math.sin(self.t * 0.1)
        biomass = 0.003 + 0.0005 * math.sin(self.t * 0.05)

        # 외란 publish
        dist_payload = {
            "radiation": rad * 200.0,          # 0~200 W/m²
            "outside_temp": out_temp,          # 20~25°C
            "outside_humidity": out_hum,       # 60~70%
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

    def run(self):
        """시뮬레이터 루프"""
        try:
            while True:
                self._simulate_day_cycle()
                time.sleep(1)  # 1초마다 전송 (speed_factor 적용)
                self.t += self.speed_factor  # 1분씩 경과 (60배속)
        except KeyboardInterrupt:
            print("🛑 Simulation stopped.")
            self.client.loop_stop()
