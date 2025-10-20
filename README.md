# Smart Greenhouse RL-MPC Control

이 프로젝트는 [“Reinforcement Learning-Based Model Predictive Control for Greenhouse Climate Control”](https://www.sciencedirect.com/science/article/pii/S2772375524003551) 논문에 기반한 RL+MPC 하이브리드 제어 시스템을 실제 스마트팜 환경에 적용하기 위해 개발되었습니다.  

본 프로젝트는 논문과 관련 GitHub 코드([SamuelMallick/mpcrl-greenhouse](https://github.com/SamuelMallick/mpcrl-greenhouse))를 참고하였으며, 시뮬레이션뿐 아니라 실제 하드웨어 환경에서도 동작하도록 수정 및 확장되었습니다.

---

## 프로젝트 개요

- **목표**: 강화학습(RL)과 모델 예측 제어(MPC)를 결합하여 온실 내 환경(온도, 습도 등)을 최적화
- **주요 구성**
  - RL Agent: Raspberry Pi에서 강화학습 수행, MPC 파라미터 학습 및 업데이트
  - MPC Controller: 환경 상태 예측 후 액추에이터 제어 명령 생성
  - 센서 및 액추에이터: ESP32와 UART/MQTT 연결, 온습도 센서, 환기팬, 모터, LED 등
- **주요 특징**
  - 실제 온실 환경에서 센서 데이터를 기반으로 RL-MPC 제어 수행
  - 시뮬레이션 코드와 실제 환경 코드 분리
  - MPC 예측 Horizon, 제약 조건, 제어 비용 등을 실제 환경에 맞게 조정
  - MQTT/UART 기반 실시간 데이터 통신 및 제어

---

## 설치 및 실행

### 요구 사항

- Python >= 3.10
- Raspberry Pi OS (혹은 Linux 환경)
- ESP32 개발 환경 (PlatformIO)
- 필요 라이브러리
```bash
pip install gymnasium casadi numpy matplotlib paho-mqtt
