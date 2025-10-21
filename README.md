# 🌿 MPCRL – Real-world MPC Controller for Smart Greenhouse

## 🧩 프로젝트 개요
이 프로젝트는 **Model Predictive Control (MPC)** 기반의 **스마트팜(온실) 제어 시스템**을 실제 환경에서 동작하도록 구현한 버전입니다.  
기존의 시뮬레이션 중심 **MPCRL (MPC + Reinforcement Learning)** 구조에서 **강화학습(RL)** 부분을 제거하고,  
**실제 센서 데이터를 기반으로 실시간 제어 가능한 MPC 시스템**만 남겨놓은 구조입니다.

---

## 🏗️ 프로젝트 구조

mpcrl_real/
├── main.py # 메인 제어 루프 (센서 → MPC → 액추에이터)
├── learning_real.py # 실제환경용 Learning MPC 정의
├── real_env.py # 실제 센서 데이터 정규화 및 액추에이터 인터페이스
├── greenhouse/
│ └── model.py # Van Henten (1994) 기반 온실 물리 모델
├── sims/configs/
│ ├── default.py # 시뮬레이션용 학습 설정 (RL 포함)
│ └── default_real.py # 실제환경용 최소 설정 (RL 제거)
└── README.md

yaml
코드 복사

---

## ⚙️ 주요 구성 요소

### 1️⃣ `main.py` – 메인 루프

실제 환경에서 **센서 → MPC 계산 → 액추에이터 제어** 흐름을 수행합니다.

```python
env = RealEnvironment(sample_time=5.0)
mpc = LearningMpcReal(test=DefaultReal())

while True:
    x_current = env.read_sensors()
    d_current = env.read_disturbance()
    u_opt, status = mpc.compute_control(x_current, d_current)
    env.apply_control(u_opt)
    env.wait_next_cycle()
센서 데이터 읽기: read_sensors()

MPC 계산: LearningMpcReal.compute_control()

액추에이터 출력: apply_control()

루프 주기: 5초(default)

2️⃣ learning_real.py – 실제환경용 MPC 제어기
CasADi와 csnlp 기반의 MPC 최적화 문제 정의 클래스

물리 모델: Model.rk4_step() / Model.euler_step()

목적 함수: 입력비용 + 상태 편차 + 제약 위반 패널티

제약조건: 출력 제약 / 입력 변화율 제한

Solver: IPOPT (max_iter=500, tol=1e-4)

python
코드 복사
u_opt, status = mpc.compute_control(x_current, d_current)
출력:

ini
코드 복사
u_opt = [fan, heater, led]
status = "Solve_Succeeded" or "Solve_Failed"
3️⃣ real_env.py – 실제 환경 인터페이스
센서 데이터를 모델 입력 스케일(0~1) 로 정규화하고,
MPC 결과를 실제 장치 제어 신호로 변환하는 클래스입니다.

🔹 센서 입력 구조
python
코드 복사
x = [biomass, humidity, temperature, leaf_water]
d = [radiation, co2, outside_temp, outside_humidity]
🔹 정규화 기준
항목	범위	정규화식
온도	15~35°C	(x - 15) / 20
습도	0~100%	x / 100
CO₂	400~2000 ppm	(x - 400) / 1600
복사량	0~200 W/m²	x / 200

🔹 제어 출력 예시
ini
코드 복사
[ACTUATOR] Fan=0.120, Heater=0.000, LED=0.350
실제 환경에서는 이 값을 PWM, 릴레이, MQTT 등으로 변환하여 적용할 수 있습니다.

4️⃣ model.py – 온실 물리 모델
Van Henten (1994) 논문 기반 상추 재배용 greenhouse 비선형 모델

State (x): biomass, humidity, temperature, leaf water

Control (u): fan, heater, LED

Disturbance (d): radiation, CO₂, outside temperature, humidity

주요 함수:

python
코드 복사
Model.df()          # 연속 시간 미분방정식
Model.rk4_step()    # Runge-Kutta 적분
Model.output()      # 출력 함수
Model.get_u_max()   # 입력 제한
Model.get_output_max()  # 출력 제한
5️⃣ default_real.py – 실제 환경용 설정값
RL 관련 항목 제거 후, 실시간 제어 안정성 중심의 최소 설정

python
코드 복사
learnable_pars_init = {
    "V0": 0,
    "c_dy": 10,
    "w": 1e3 * np.ones((4,)),
    "c_y": 10,
    "y_fin": 26,
    "c_u": [1, 0.5, 0.5],
}
파라미터	의미
c_u	제어 입력비용 (낮을수록 적극 제어)
c_y	목표 상태 도달 강도
y_fin	목표 온도/출력값
w	제약조건 위반 패널티
c_dy	출력 변화율 가중치

🚀 실행 방법
🔧 1. 환경 세팅
bash
코드 복사
git clone https://github.com/HyunwooKim01/mpcrl.git
cd mpcrl
pip install -r requirements.txt
(필요시 추가 패키지 설치)

bash
코드 복사
pip install casadi csnlp mpcrl numpy
▶️ 2. 실행
bash
코드 복사
python main.py
실행 예시:

csharp
코드 복사
🚀 Real-world MPC controller starting...
✅ MPC controller initialized.
[SENSOR] x=[0.003 0.540 0.680 0.008], d=[0.360 0.410 0.650 0.600]
[MPC] status=Solve_Succeeded, u_opt=[0.1 0.0 0.2]
[ACTUATOR] Fan=0.100, Heater=0.000, LED=0.200
🧠 핵심 설계 포인트
구성 요소	역할
LearningMpcReal	실시간 MPC 최적화 수행 (CasADi 기반)
RealEnvironment	센서-액추에이터 인터페이스 및 정규화
Model	Van Henten greenhouse 물리 모델
DefaultReal	실제 제어 환경 초기 파라미터 정의

🧩 확장 계획
기능	설명
✅ MQTT 통신 연동	스마트팜 서버와 제어 신호 교환
✅ 센서 실데이터 연동	온도/습도/CO₂ 센서 값 직접 반영
⚙️ RL Fine-Tuning	RL을 통한 MPC 정책 보정 (MPCRL 완전 통합)
☁️ 클라우드 로깅	TimescaleDB 기반 제어 데이터 수집 및 분석

📚 참고 논문
Van Henten, E. J. (1994). Greenhouse climate control: an integrated approach.
“Reinforcement Learning-based Model Predictive Control for Greenhouse Climate Control,” ScienceDirect, 2024.

🏁 요약
이 프로젝트는 실제 스마트팜 환경에서 MPC 기반 제어를 실시간으로 수행하기 위한 경량화된 MPCRL 구조입니다.
센서 입력만으로 제어 결정을 수행하며, 추후 RL Fine-Tuning을 결합해 완전한 On-Device AI Controller로 발전할 수 있습니다.

✨ 개발자 메모
본 프로젝트는 학습 중심의 learning_mpc.py 구조를 기반으로, 실제 환경에서 바로 적용 가능한 형태로 단순화되었습니다.
향후 ESP32·Raspberry Pi 등 임베디드 시스템에 이식 시, RealEnvironment 인터페이스만 수정하면 바로 활용 가능합니다.
