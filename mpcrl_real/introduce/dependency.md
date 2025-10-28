⚙️ 의존 관계 다이어그램


┌──────────────────────────────────┐
│ q_learning_greenhouse_pi.py      │
│  • RL(TD update) 수행            │
│  • θ 업데이트 / 저장               │
└────────────┬─────────────────────┘
             │ uses
             ▼
┌────────────────────────┐
│ learning_real_rl.py    │
│  • MPC + RL Hook       │
│  • compute_control()   │
│  • eval_Q/V(), reward()│
└────────┬───────────────┘
         │
         ├────▶ greenhouse/model_real.py
         │         (온도·습도·생체량 모델)
         │
         ├────▶ sims/configs/default_real.py
         │         (초기 가중치, crop_profile 연동)
         │
         └────▶ mqtt_handler.py
                   (센서 수신 / 제어 명령 송신)


| 역할                | 입력            | 출력         | 사용 파일                         |
| ----------------- | ------------- | ---------- | ----------------------------- |
| MQTT Handler      | 센서 MQTT 메시지   | 제어 MQTT 명령 | `mqtt_handler.py`             |
| MPC Controller    | x,d 센서상태      | u 제어입력     | `learning_real_rl.py`         |
| RL Loop           | MPC 내부 cost θ | 업데이트된 θ    | `q_learning_greenhouse_pi.py` |
| Main Control Loop | 센서→MPC→액추에이터  | 실시간 제어     | `main.py`                     |
