#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import threading
from typing import Dict
import paho.mqtt.client as mqtt
import numpy as np

from env import LettuceGreenHouse
from learning import LearningMpc  # 반드시 LearningMpc 구현 필요

# ---------------- MQTT 설정 ----------------
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
MQTT_QOS = 1
CLIENT_ID = "rpi-greenhouse-mpc"

TOPIC_SENSOR_COMBINED = "farm/greenhouse/sensors"
TOPIC_ACTUATORS = {
    "heater": "farm/greenhouse/actuators/heater",
    "humidifier": "farm/greenhouse/actuators/humidifier",
    "co2_valve": "farm/greenhouse/actuators/co2_valve",
    "led": "farm/greenhouse/actuators/led"
}
TOPIC_STATE_UPDATE = "farm/greenhouse/mpc/state"
TOPIC_THETA_UPDATE = "farm/greenhouse/mpc/theta_update"

CONTROL_PERIOD_SEC = 60
STATE_KEYS = ["temp","hum","co2","light"]
Y_BOUNDS = {
    "temp":  (18,24),
    "hum":   (55,75),
    "co2":   (400,900),
    "light": (150,500)
}

# ---------------- 내부 상태 ----------------
latest_state: Dict[str,float] = {}
state_lock = threading.Lock()
theta = {}  # MPC 파라미터

# ---------------- MQTT 콜백 ----------------
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected rc={rc}")
    client.subscribe(TOPIC_SENSOR_COMBINED, qos=MQTT_QOS)
    client.subscribe(TOPIC_THETA_UPDATE, qos=MQTT_QOS)

def on_message(client, userdata, msg):
    global latest_state, theta, mpc
    topic = msg.topic
    payload = msg.payload.decode("utf-8")
    try:
        data = json.loads(payload)
        if topic == TOPIC_SENSOR_COMBINED:
            with state_lock:
                for k in STATE_KEYS:
                    if k in data:
                        latest_state[k] = float(data[k])
        elif topic == TOPIC_THETA_UPDATE:
            theta.update(data)  # Fine-Tuning 결과 반영
            for k,v in data.items():
                if hasattr(mpc, "set_value"):
                    mpc.set_value(k,v)
    except Exception as e:
        print(f"[MQTT] parse error: {e}")

# ---------------- 상태 벡터 생성 ----------------
def build_state_vector():
    with state_lock:
        try:
            return [latest_state[k] for k in STATE_KEYS]
        except KeyError:
            return None

# ---------------- 액추에이터 퍼블리시 ----------------
def publish_actuators(client, cmd:Dict[str,float]):
    ts = int(time.time())
    for name,val in cmd.items():
        topic = TOPIC_ACTUATORS.get(name)
        if topic:
            payload = json.dumps({"value":float(val),"ts":ts})
            client.publish(topic,payload=payload,qos=MQTT_QOS,retain=False)
    # 상태 토픽도 퍼블리시 (Fine-Tuning용)
    payload = json.dumps({"x": latest_state, "u": cmd})
    client.publish(TOPIC_STATE_UPDATE, payload=payload, qos=MQTT_QOS, retain=False)

# ---------------- MPC 실행 ----------------
class MpcRunner:
    def __init__(self):
        global mpc, theta
        # 환경 객체 생성
        greenhouse_env = LettuceGreenHouse(
            growing_days=30,   # 필요에 따라 재배 일수 조정
            model_type="rk4",
            disturbance_profiles_type="single",
        )

        # theta_params.json 로드
        with open("theta_params.json","r") as f:
            theta = json.load(f)["params"]

        mpc = LearningMpc(
            greenhouse_env=greenhouse_env,
            test=None,
            np_random=np.random,
            prediction_horizon=24,   # 6시간 * 4step/h
            prediction_model="rk4",
            constrain_control_rate=True
        )

        for k,v in theta.items():
            mpc.set_value(k,v)
        print("[MPC] 초기화 완료")

    def step(self, x:list) -> Dict[str,float]:
        # 출력 범위, 외란 세팅
        for k, key in enumerate(STATE_KEYS):
            ymin,ymax = Y_BOUNDS[key]
            mpc.set_value(f"y_min_{k}", [[ymin]*len(STATE_KEYS)])
            mpc.set_value(f"y_max_{k}", [[ymax]*len(STATE_KEYS)])
        mpc.set_value("d", [[0.0]*mpc.prediction_horizon for _ in STATE_KEYS])
        try:
            u = mpc.solve(x)
        except Exception:
            u = self._fallback_policy(x)
        cmd = {
            "heater": float(u[0]) if len(u)>0 else 0.0,
            "humidifier": float(u[1]) if len(u)>1 else 0.0,
            "co2_valve": float(u[2]) if len(u)>2 else 0.0,
            "led": float(u[3]) if len(u)>3 else 0.0
        }
        return cmd

    def _fallback_policy(self,x):
        setpoints = [sum(Y_BOUNDS[k])/2 for k in STATE_KEYS]
        kp = [0.1,0.1,0.05,0.05]
        u = [kp[i]*(setpoints[i]-x[i]) for i in range(len(setpoints))]
        while len(u)<4: u.append(0.0)
        return u

# ---------------- 메인 루프 ----------------
def main():
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST,BROKER_PORT,60)
    client.loop_start()
    client.publish("farm/greenhouse/rpi/status",json.dumps({"status":"online"}),qos=MQTT_QOS,retain=True)

    runner = MpcRunner()
    try:
        while True:
            x = build_state_vector()
            if x is None:
                time.sleep(1)
                continue
            cmd = runner.step(x)
            publish_actuators(client, cmd)
            print(f"x={x} -> cmd={cmd}")
            time.sleep(CONTROL_PERIOD_SEC)
    except KeyboardInterrupt:
        client.loop_stop()
        client.disconnect()
        print("[MPC] 종료")

if __name__=="__main__":
    main()
