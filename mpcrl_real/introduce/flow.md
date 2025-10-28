[ESP32 Sensor Node] ─(MQTT)─▶ [mqtt_handler.py]
                               │
                               ▼
                        [LearningMpcReal]
                               │
                               ▼
                       compute_control()
                               │
                               ▼
                       [publish_actuator()]
                               │
                               └─▶ MQTT → ESP32 제어신호
