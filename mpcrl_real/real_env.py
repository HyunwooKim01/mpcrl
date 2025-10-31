# -*- coding: utf-8 -*-
"""
real_env.py
실제 환경과 RL-MPC 사이를 연결하는 I/O 어댑터
"""

from __future__ import annotations
import json, time, threading
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import paho.mqtt.client as mqtt


# ───────────────────────────────────────────────────────
# 환경 설정
# ───────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    broker_host: str = "172.27.148.207"
    broker_port: int = 1883
    farm_id: str = "farmA"
    esp_id: str = "esp1"
    sample_time: float = 5.0
    use_normalization: bool = False
    crop_profile_path: str = "crops/lettuce_profile.json"
    sensor_timeout: float = 10.0


def _g(d: Dict, k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return default


# ───────────────────────────────────────────────────────
# 실제 환경 클래스
# ───────────────────────────────────────────────────────
class RealEnvironment:
    def __init__(self, sample_time: float = 5.0, **kwargs):
        self.cfg = EnvConfig(sample_time=sample_time, **kwargs)
        self.topics = {
            "sensor": f"{self.cfg.farm_id}/{self.cfg.esp_id}/sensor",
            "disturbance": f"{self.cfg.farm_id}/{self.cfg.esp_id}/disturbance",
            "actuator_control": f"{self.cfg.farm_id}/{self.cfg.esp_id}/actuator/control",
        }

        self._lock = threading.RLock()
        now = time.time()
        self._sensor = {
            "temp_in": 0.0,
            "hum_in": 0.0,
            "co2_in": 0.0,
            "light_in": 0.0,
            "timestamp": now,
        }
        self._dist = {
            "solar_rad": 0.0,
            "co2_out": 420.0,
            "temp_out": 15.0,
            "hum_out": 50.0,
            "timestamp": now,
        }

        self.crop = self._load_crop_profile(self.cfg.crop_profile_path)

        # ───────────────────────────────
        # MQTT 클라이언트 설정
        # ───────────────────────────────
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)

        try:
            self.mqtt_client.connect(self.cfg.broker_host, self.cfg.broker_port, keepalive=30)
            print(f"✅ MQTT connected → {self.cfg.broker_host}:{self.cfg.broker_port}")
        except Exception as e:
            print(f"❌ MQTT connection failed: {e}")
            print("💡 Mosquitto가 실행 중인지 확인하세요.")

        self.mqtt_client.loop_start()
        time.sleep(1)
        print(f"📡 Subscribed topics: {self.topics}")

        self.sample_time = self.cfg.sample_time

    # ──────────────────────────────────────────────
    # MQTT 콜백
    # ──────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self.topics["sensor"], qos=0)
            client.subscribe(self.topics["disturbance"], qos=0)
            print(f"🟢 MQTT connected successfully, subscribed to {self.topics}")
        else:
            print(f"❌ MQTT connect failed (rc={rc})")

    def _on_message(self, client, userdata, msg):
        try:
            payload_str = msg.payload.decode("utf-8", errors="ignore")
            payload = json.loads(payload_str)
            print(f"📩 MQTT {msg.topic} : {payload}")
        except Exception as e:
            print(f"⚠️ decode error: {e}")
            return

        now = time.time()
        with self._lock:
            if msg.topic == self.topics["sensor"]:
                self._sensor.update({
                    "temp_in": _g(payload, "temp_in", self._sensor["temp_in"]),
                    "hum_in": _g(payload, "hum_in", self._sensor["hum_in"]),
                    "co2_in": _g(payload, "co2_in", self._sensor["co2_in"]),
                    "light_in": _g(payload, "light_in", self._sensor["light_in"]),
                    "timestamp": _g(payload, "timestamp", now),
                })
            elif msg.topic == self.topics["disturbance"]:
                self._dist.update({
                    "solar_rad": _g(payload, "solar_rad", self._dist["solar_rad"]),
                    "co2_out": _g(payload, "co2_out", self._dist["co2_out"]),
                    "temp_out": _g(payload, "temp_out", self._dist["temp_out"]),
                    "hum_out": _g(payload, "hum_out", self._dist["hum_out"]),
                    "timestamp": _g(payload, "timestamp", now),
                })

    # ──────────────────────────────────────────────
    # 데이터 읽기/쓰기
    # ──────────────────────────────────────────────
    def read_sensors(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            s = dict(self._sensor)
            w = dict(self._dist)

        now = time.time()
        if now - s["timestamp"] > self.cfg.sensor_timeout:
            print("[WARN] Sensor payload stale (> timeout).")
        if now - w["timestamp"] > self.cfg.sensor_timeout:
            print("[WARN] Disturbance payload stale (> timeout).")

        x = np.array(
            [s["temp_in"], s["hum_in"], s["co2_in"], s["light_in"]],
            dtype=float,
        )
        d = np.array(
            [w["solar_rad"], w["co2_out"], w["temp_out"], w["hum_out"]],
            dtype=float,
        )
        return x, d

    def send_actuators(self, u_opt: np.ndarray) -> None:
        fan, heater, led = [float(np.clip(u, 0.0, 1.0)) for u in u_opt]
        payload = {"fan": fan, "heater": heater, "led": led, "timestamp": time.time()}
        self.mqtt_client.publish(self.topics["actuator_control"], json.dumps(payload), qos=0)
        print(f"📤 Sent actuator command → {payload}")

    # ──────────────────────────────────────────────
    # 작물 프로필 / 정규화
    # ──────────────────────────────────────────────
    def _load_crop_profile(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return {"target_temp": [18.0, 22.0], "target_humidity": [50.0, 70.0]}
