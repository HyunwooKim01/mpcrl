# 🌿 MPCRL 프로젝트 전체 구조 개요

이 프로젝트는  
📘 **“Reinforcement Learning-Based Model Predictive Control for Greenhouse Climate Control”**  
논문 기반 구조입니다.

기본적으로 **CasADi 기반 MPC 최적화기**를 중심으로,  
**RL이 MPC의 파라미터를 학습(보정)** 하는 형태로 작동합니다.

---

## ⚙️ Ⅰ. MPC (Model Predictive Control) 실행 구조

> ✅ 학습 없이, 모델 기반 최적 제어만 수행할 때 사용되는 구조

### 📁 주요 파일 구조

| 파일 | 역할 | 상위/하위 의존성 |
|------|------|------------------|
| **learning.py** | MPC 제어기(`LearningMpc`) 클래스 정의<br>CasADi 기반 최적화 문제 구성 및 제약 조건, 비용 함수 설정 | 🔽 의존: `greenhouse/model.py`, `greenhouse/env.py`, `sims/configs/default.py` |
| **greenhouse/model.py** | 온실 시스템의 물리적 모델 정의<br>Van Henten(1994) 기반 미분방정식 | 독립적 (모든 곳에서 참조됨) |
| **greenhouse/env.py** | 시뮬레이션 환경(`LettuceGreenHouse`) 정의<br>센서 상태, 외란, 제어 입력 업데이트 | 🔽 의존: `model.py`, `utils/brownian_motion.py`, `data/disturbances.npy` |
| **sims/configs/default.py** | MPC 및 시뮬레이션 기본 설정 (horizon, discount, learning rate 등) | 독립적 |
| **utils/brownian_motion.py** | 외란 노이즈(브라운 운동) 생성 | 독립적 |
| **data/disturbances.npy** | 실제 기상 외란 데이터 (복사량, 온도, 습도 등) | 데이터 파일 (env에서 사용됨) |
| **test_80.py** | 실험용 config (DefaultTest 상속, 특정 조건 적용) | 🔽 의존: `default.py`, `model.py` |

### ⚙️ MPC 동작 흐름 요약

#### 1️⃣ Model (`model.py`)
- 상태(`x`), 제어(`u`), 외란(`d`) 기반 미분방정식 정의  
- 예: `dx = f(x, u, d, p)`

#### 2️⃣ Environment (`env.py`)
- 모델을 이용해 다음 상태 예측 (`rk4_step`)  
- 센서 → 상태 관찰, 외란 → Disturbance 프로파일 관리

#### 3️⃣ LearningMpc (`learning.py`)
- CasADi 기반 NLP(Nonlinear Programming) 생성  
- 제약조건(`y_min <= y <= y_max`), 목적함수(`obj`) 설정  
- IPOPT solver 초기화 후 매 step 제어 입력 계산

#### 4️⃣ Config (`default.py`, `test_80.py`)
- Horizon, learning rate, 제약 가중치 등 설정 전달

### 🔗 MPC 의존 관계도

```
learning.py (LearningMpc)
│
├── greenhouse/env.py (LettuceGreenHouse)
│    ├── greenhouse/model.py (Model)
│    ├── utils/brownian_motion.py
│    └── data/disturbances.npy
│
├── sims/configs/default.py
│    └── greenhouse/model.py
└── casadi, csnlp, numpy
```

### 💡 MPC만 실행할 때 필요한 파일

#### ✅ 필수 파일
```
learning.py
greenhouse/env.py
greenhouse/model.py
sims/configs/default.py
utils/brownian_motion.py
data/disturbances.npy
test_80.py
```

#### ✅ 필수 라이브러리
```
casadi
csnlp
numpy
gymnasium
```

---

## 🧠 Ⅱ. RL (MPC + Reinforcement Learning) 통합 실행 구조

> ✅ 강화학습을 통해 MPC의 파라미터(`p`, `c_u`, `c_dy`, `w` 등)를 학습  
> 즉, MPC가 제어는 하고 RL이 학습(보정)을 담당

### 📁 주요 파일 구조

| 파일 | 역할 | 상위/하위 의존성 |
|------|------|------------------|
| **q_learning_greenhouse.py** | 메인 학습 스크립트<br>MPC + RL 통합 루프 실행 (학습, 평가, 저장, 시각화) | 🔽 의존: 모든 파일 |
| **greenhouse_agent.py** | RL 에이전트 정의 (`GreenhouseLearningAgent`)<br>`mpcrl.LstdQLearningAgent` 상속 | 🔽 의존: `greenhouse/env.py`, `greenhouse/model.py`, `mpcrl` |
| **learning.py** | MPC 제어기 (`LearningMpc`) | 🔽 의존: `model.py`, `env.py`, `default.py` |
| **greenhouse/env.py** | RL 환경 (`gym.Env` 호환) | 🔽 의존: `model.py`, `brownian_motion.py` |
| **greenhouse/model.py** | 물리 모델 | 독립적 |
| **sims/configs/test_80.py** | RL 실험 설정 (학습률, step 수 등) | 🔽 의존: `default.py`, `model.py` |
| **utils/plot.py** | 학습 결과 시각화 (TD-error, 보상, 온실 상태 등) | 🔽 의존: `env.py`, `model.py`, `matplotlib` |

### 🔁 RL 실행 흐름 요약

1️⃣ **환경 초기화 (`LettuceGreenHouse`)**  
외란, 초기 상태, 제어 입력 제한 설정

2️⃣ **MPC 생성 (`LearningMpc`)**  
예측 모델, 제약조건, 목적함수 설정  
파라미터(`p`, `c_u`, `w`, `y_fin`)를 학습 가능한 변수로 설정

3️⃣ **에이전트 생성 (`GreenhouseLearningAgent`)**  
- `mpcrl.LstdQLearningAgent` 기반  
- MPC 파라미터를 RL로 학습  
- TD Error, Exploration, Experience Replay 관리

4️⃣ **훈련 루프 (`q_learning_greenhouse.py`)**  
- Episode 단위 시뮬레이션  
- `agent.train()`으로 RL + MPC 결합 학습 수행  
- `.pkl` 파일로 결과 저장  
- `plot_greenhouse()`로 시각화

### 🔗 RL 의존 관계도

```
q_learning_greenhouse.py
│
├── agents/greenhouse_agent.py
│    ├── mpcrl (LstdQLearningAgent, Agent)
│    ├── greenhouse/env.py
│    └── greenhouse/model.py
│
├── learning.py (LearningMpc)
│    ├── greenhouse/model.py
│    ├── sims/configs/test_80.py
│    └── casadi, csnlp
│
├── utils/plot.py (시각화)
│    ├── greenhouse/env.py
│    └── matplotlib
│
└── mpcrl, numpy, gymnasium, pickle
```

### 💡 RL(MPC+RL) 실행 시 필요한 파일

#### ✅ 필수 파일
```
q_learning_greenhouse.py
agents/greenhouse_agent.py
learning.py
greenhouse/env.py
greenhouse/model.py
sims/configs/test_80.py
sims/configs/default.py
utils/brownian_motion.py
utils/plot.py
data/disturbances.npy
```

#### ✅ 필수 라이브러리
```
numpy
matplotlib
gymnasium
casadi
csnlp
mpcrl
```

---

## 🔍 MPC ↔ RL 의존성 관계 핵심 요약

| 구성 | 역할 | 연결 대상 | 설명 |
|------|------|-----------|------|
| **Model** | 물리 시스템 모델 | Environment, LearningMpc, Agent | 시스템 동특성 계산 |
| **Environment** | 시뮬레이션 환경 | Agent, LearningMpc | 상태/보상 반환 |
| **LearningMpc** | 최적 제어 계산 | Agent, q_learning_greenhouse | 제어 입력 생성 |
| **Agent** | 학습 주체 | LearningMpc, Environment | RL로 파라미터 학습 |
| **Config(Test)** | 학습 및 시뮬레이션 설정 | LearningMpc, Agent | 하이퍼파라미터 관리 |
| **Plot** | 결과 시각화 | q_learning_greenhouse | 보상, TD-error 표시 |

---

## 🎯 정리 요약

| 목적 | 필요한 파일 | 설명 |
|------|--------------|------|
| 🔹 **MPC 제어 테스트** | `learning.py`, `model.py`, `env.py`, `default.py`, `brownian_motion.py`, `disturbances.npy` | 모델 예측 기반 제어 시뮬레이션 |
| 🔹 **RL 학습(MPC+RL)** | 위 + `q_learning_greenhouse.py`, `greenhouse_agent.py`, `plot.py`, `test_80.py` | RL로 MPC 파라미터 학습 및 평가 |
| 🔹 **공통 데이터/모듈** | `model.py`, `env.py`, `data/disturbances.npy` | 시스템 상태/외란 관리 |
| 🔹 **공통 라이브러리** | `casadi`, `csnlp`, `mpcrl`, `gymnasium`, `numpy`, `matplotlib` | 최적화 + RL 프레임워크 |
