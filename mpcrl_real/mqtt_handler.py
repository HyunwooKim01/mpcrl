# mqtt_handler.py
import paho.mqtt.client as mqtt
import json, time

BROKER_IP = "localhost"          # 또는 라즈베리파이 IP
SENSOR_TOPIC = "farmA/esp1/sensor/value"
ACTUATOR_TOPIC = "farmA/esp1/actuator/control"

latest_sensor = None

# ────────────────────────────────────────────────
# MQTT 콜백
# ────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(SENSOR_TOPIC)

def on_message(client, userdata, msg):
    global latest_sensor
    try:
        payload = msg.payload.decode("utf-8")
        latest_sensor = json.loads(payload)
    except Exception as e:
        print(f"[MQTT] Message parse error: {e}")

# ────────────────────────────────────────────────
# MQTT 초기화
# ────────────────────────────────────────────────
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER_IP, 1883, 60)
client.loop_start()

# ────────────────────────────────────────────────
# 데이터 접근/제어
# ────────────────────────────────────────────────
def get_latest_sensor():
    """가장 최근 수신된 센서 데이터 반환 (없으면 None)"""
    global latest_sensor
    return latest_sensor

def publish_actuator(u_vec):
    """액추에이터 제어 MQTT Publish"""
    try:
        payload = {
            "fan": float(u_vec[0]),
            "heater": float(u_vec[1]),
            "led": float(u_vec[2]),
            "timestamp": time.time(),
        }
        client.publish(ACTUATOR_TOPIC, json.dumps(payload))
        print(f"[MQTT→ESP] {payload}")
    except Exception as e:
        print(f"[MQTT Publish Error] {e}")
