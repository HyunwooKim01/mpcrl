🌿 RL-MPC Greenhouse Control (Raspberry Pi Ver.)
📦 프로젝트 디렉토리 구조

mpcrl_real/
│
├── main.py                        # 실시간 RL-MPC 제어 루프 (MQTT 기반)
├── q_learning_greenhouse_pi.py    # Raspberry Pi 경량 RL 학습 루프 (TD 기반)
├── mqtt_handler.py                # MQTT 통신: 센서 구독 + 제어 명령 발행
│
├── learning_real_rl.py            # RL-통합형 MPC 클래스 (CasADi 기반)
│
├── sims/
│   └── configs/
│       └── default_real.py        # 작물 프로필 로드 + 초기 MPC 파라미터 생성
│
├── greenhouse/
│   └── model_real.py              # 실환경용 단순화된 온실 물리모델(df, rk4 등)
│
├── crops/
│   └── lettuce_profile.json       # 상추 작물 프로필 (목표온도, 가중치 등)
│
└── logs/
    └── rl_mpc_log.csv             # (자동 생성) RL 학습 로그 저장
