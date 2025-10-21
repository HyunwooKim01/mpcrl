"""
real_env.py
------------
실제환경용 환경 클래스.
ESP32 ↔ MQTT 브로커 ↔ 라즈베리파이 간 실시간 데이터 송수신을 담당한다.
센서값은 ESP에서 MQTT로 들어오고,
MPC 계산 결과(u_opt)는 다시 MQTT로 ESP로 전송된다.
"""

import time
import json
import numpy as np
import paho.mqtt.client as mqtt


class RealEnvironment:
    def __init__(self,
                 broker_ip: str = "localhost",
                 farm_id: str = "farmA",
                 esp_id: str = "esp1",
                 sample_time: float = 5.0):
        """
        broker_ip  : MQTT 브로커 주소 (예: "192.168.0.10")
        farm_id    : 농장 이름 (MQTT 토픽 prefix)
        esp_id     : ESP 장치 ID
        sample_time: 제어 주기 (초)
        """
        self.sample_time = sample_time
        self.sensor_data = {}
        self.disturbance_data = {}

        # MQTT 토픽 구성
        self.topic_sensor = f"{farm_id}/{esp_id}/sensor"
        self.topic_disturbance = f"{farm_id}/{esp_id}/disturbance"
        self.topic_actuator = f"{farm_id}/{esp_id}/actuator/control"

        # MQTT 클라이언트 설정
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        print(f"🔗 Connecting to MQTT Broker at {broker_ip}...")
        self.client.connect(broker_ip, 1883, 60)
        self.client.loop_start()

    # ---------------- MQTT 이벤트 핸들러 ----------------
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT connected successfully.")
            # 센서 및 외란 데이터 구독
            self.client.subscribe(self.topic_sensor)
            self.client.subscribe(self.topic_disturbance)
            print(f"📡 Subscribed to: {self.topic_sensor}, {self.topic_disturbance}")
        else:
            print(f"❌ MQTT connection failed (code {rc})")

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode("utf-8"))
        topic = msg.topic
        if "sensor" in topic:
            self.sensor_data = payload
        elif "disturbance" in topic:
            self.disturbance_data = payload

    # ---------------- 센서 입력 ----------------
    def read_sensors(self) -> np.ndarray:
        """
        ESP에서 받은 내부 센서 데이터 정규화
        - biomass (0~0.005)
        - humidity (0~100%)
        - temperature (15~35°C)
        - leaf_water (0.006~0.009)
        """
        if not self.sensor_data:
            return np.zeros(4)
        s = self.sensor_data
        temp_norm = (s.get("temperature", 25.0) - 15.0) / 20.0
        hum_norm = s.get("humidity", 60.0) / 100.0
        biomass = s.get("biomass", 0.003)
        leaf_water = s.get("leaf_water", 0.007)
        return np.array([biomass, hum_norm, temp_norm, leaf_water])

    # ---------------- 외란 입력 ----------------
    def read_disturbance(self) -> np.ndarray:
        """
        외부 환경 정보 (복사량, 외기온, 외기습도, 낮/밤)
        """
        if not self.disturbance_data:
            return np.zeros(4)
        d = self.disturbance_data
        rad_norm = d.get("radiation", 100.0) / 200.0
        out_temp_norm = (d.get("outside_temp", 25.0) - 15.0) / 20.0
        out_hum_norm = d.get("outside_humidity", 60.0) / 100.0
        hour = time.localtime().tm_hour
        time_norm = 1.0 if 6 <= hour <= 18 else 0.0
        return np.array([rad_norm, out_temp_norm, out_hum_norm, time_norm])

    # ---------------- 제어 출력 ----------------
    def apply_control(self, u_opt: np.ndarray):
        """
        MPC 계산 결과를 ESP로 MQTT Publish
        """
        payload = json.dumps({
            "fan": round(float(u_opt[0]), 3),
            "heater": round(float(u_opt[1]), 3),
            "led": round(float(u_opt[2]), 3),
            "timestamp": time.time()
        })
        self.client.publish(self.topic_actuator, payload)
        print(f"[MQTT→ESP] {self.topic_actuator} : {payload}")

    # ---------------- 루프 주기 대기 ----------------
    def wait_next_cycle(self):
        time.sleep(self.sample_time)
