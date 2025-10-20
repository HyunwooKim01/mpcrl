🌿 MPCRL 프로젝트 전체 구조 개요

이 프로젝트는
📘 “Reinforcement Learning-Based Model Predictive Control for Greenhouse Climate Control”
논문 기반 구조입니다.

기본적으로 CasADi 기반 MPC 최적화기를 중심으로,
RL이 MPC의 파라미터를 학습(보정) 하는 형태로 작동합니다.

| 파일                             | 역할                                                                     | 상위/하위 의존성                                                                    |
| ------------------------------ | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **`learning.py`**              | MPC 제어기(`LearningMpc`) 클래스 정의<br>CasADi 기반 최적화 문제 구성 및 제약 조건, 비용 함수 설정 | 🔽 의존: `greenhouse/model.py`, `greenhouse/env.py`, `sims/configs/default.py` |
| **`greenhouse/model.py`**      | 온실 시스템의 물리적 모델 정의<br>Van Henten(1994) 기반 미분방정식                         | 독립적 (모든 곳에서 참조됨)                                                             |
| **`greenhouse/env.py`**        | 시뮬레이션 환경(`LettuceGreenHouse`) 정의<br>센서 상태, 외란, 제어 입력 업데이트              | 🔽 의존: `model.py`, `utils/brownian_motion.py`, `data/disturbances.npy`       |
| **`sims/configs/default.py`**  | MPC 및 시뮬레이션 기본 설정 (horizon, discount, learning rate 등)                 | 독립적                                                                          |
| **`utils/brownian_motion.py`** | 외란 노이즈(브라운 운동) 생성                                                      | 독립적                                                                          |
| **`data/disturbances.npy`**    | 실제 기상 외란 데이터 (복사량, 온도, 습도 등)                                           | 데이터 파일 (env에서 사용됨)                                                           |
| **`test_80.py`**               | 실험용 config (DefaultTest 상속, 특정 조건 적용)                                  | 🔽 의존: `default.py`, `model.py`                                              |
