#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from real_learning import RealLearningMpc  # 새로 만든 RealLearningMpc
from real_env import RealSensorEnv  # 실제 센서 환경

# ---------------- MQTT 설정 ----------------
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
CLIENT_ID = "rpi-greenhouse-mpc"
TOPIC_SENSOR_COMBINED = "farm/greenhouse/sensor/combined"
TOPIC_ACTUATOR_CMD = "farm/greenhouse/actuator/cmd"

CONTROL_PERIOD_SEC = 15  # 제어 주기 15초

# ---------------- 환경 및 MPC 초기화 ----------------
env = RealSensorEnv()

# 실제 환경용 MPC
mpc = RealLearningMpc(
    greenhouse_env=None,  # 실제 센서 기반
    test=None,
    prediction_horizon=6*4,  # 6시간 horizon
    prediction_model="rk4",
    constrain_control_rate=True,
)

# ---------------- MQTT 콜백 ----------------
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected rc={rc}")
    client.subscribe(TOPIC_SENSOR_COMBINED, qos=1)

def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")
    try:
        data = json.loads(payload)
        env.set_state(data)  # 센서 데이터를 환경에 반영
    except Exception as e:
        print(f"[MQTT] parse error: {e}")

# ---------------- 액추에이터 퍼블리시 ----------------
def publish_actuators(client, action):
    """MPC가 계산한 명령을 MQTT로 발행"""
    action_dict = {
        "heater": float(action[0]),
        "humidifier": float(action[1]),
        "co2_valve": float(action[2]),
        "led": float(action[3]),
    }
    client.publish(TOPIC_ACTUATOR_CMD, json.dumps(action_dict), qos=1)
    print(f"[ACT] {action_dict}")

# ---------------- 메인 루프 ----------------
def main():
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    try:
        while True:
            x = env.get_state_vector()
            if x is not None:
                # 실제 센서 값 기반으로 MPC 제어
                u = mpc.step(x)
                publish_actuators(client, u)
            time.sleep(CONTROL_PERIOD_SEC)
    except KeyboardInterrupt:
        print("[MPC] 종료")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
