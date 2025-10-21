# 🌿 MPCRL – Real-world MPC Controller for Smart Greenhouse

## 🧩 프로젝트 개요
이 프로젝트는 **Model Predictive Control (MPC)** 기반의 **스마트팜(온실) 제어 시스템**을 실제 환경에서 동작하도록 구현한 버전입니다.  
기존의 시뮬레이션 중심 **MPCRL (MPC + Reinforcement Learning)** 구조에서 **강화학습(RL)** 부분을 제거하고,  
**실제 센서 데이터를 기반으로 실시간 제어 가능한 MPC 시스템**만 남겨놓은 구조입니다.

---

## 🏗️ 프로젝트 구조

```plaintext
mpcrl_real/
├── main.py                 # 메인 제어 루프 (센서 → MPC → 액추에이터)
├── learning_real.py        # 실제환경용 Learning MPC 정의
├── real_env.py             # 실제 센서 데이터 정규화 및 액추에이터 인터페이스
├── greenhouse/
│   └── model.py            # Van Henten (1994) 기반 온실 물리 모델
├── sims/configs/
│   ├── default.py          # 시뮬레이션용 학습 설정 (RL 포함)
│   └── default_real.py     # 실제환경용 최소 설정 (RL 제거)
└── README.md
