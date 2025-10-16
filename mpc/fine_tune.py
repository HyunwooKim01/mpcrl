#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import paho.mqtt.client as mqtt

FINE_TUNE_LR = 0.01  # 학습률
TOPIC_STATE_UPDATE = "farm/greenhouse/mpc/state"
TOPIC_THETA_UPDATE = "farm/greenhouse/mpc/theta_update"
CLIENT_ID = "rpi-greenhouse-finetune"

theta = {}

def on_connect(client, userdata, flags, rc):
    print(f"[FineTune] Connected rc={rc}")
    client.subscribe(TOPIC_STATE_UPDATE, qos=1)

def on_message(client, userdata, msg):
    global theta
    payload = msg.payload.decode("utf-8")
    try:
        data = json.loads(payload)
        x = [float(data["x"][k]) for k in ["temp","hum","co2","light"]]
        # 간단 Q-learning style update
        target = [21,65,650,325]  # 예시 목표값
        delta = [(t - xi)*FINE_TUNE_LR for xi,t in zip(x,target)]
        # theta 갱신
        for i,k in enumerate(theta.keys()):
            v = theta[k]
            if isinstance(v,(list,tuple)):
                theta[k] = [vi+delta[i%len(delta)] for vi in v]
            else:
                theta[k] += delta[i%len(delta)]
        # 업데이트 퍼블리시
        client.publish(TOPIC_THETA_UPDATE,json.dumps(theta),qos=1)
        print(f"[FineTune] x={x} delta={delta}")
    except Exception as e:
        print(f"[FineTune] parse error: {e}")

def main():
    client = mqtt.Client(client_id=CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("127.0.0.1",1883,60)
    client.loop_forever()

if __name__=="__main__":
    main()
