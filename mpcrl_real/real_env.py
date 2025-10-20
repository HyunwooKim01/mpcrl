"""
real_env.py
------------
실제 환경(온실, 스마트팜 등)에서 센서 데이터를 수집하고
액추에이터 제어 명령을 송신하는 모듈.
MQTT / HTTP / Serial / 직접 GPIO 기반 입력 모두 확장 가능.
"""

import time
import numpy as np
import random

class RealEnvironment:
    """실제 센서 데이터를 수집하고 액추에이터를 제어하는 클래스"""

    def __init__(self, sample_time: float = 60.0 * 15.0):
        """
        Parameters
        ----------
        sample_time : float
            제어 주기 (초 단위, default: 15분)
        """
        self.sample_time = sample_time

    # ------------------- 센서 입력 -------------------
    def read_sensors(self) -> np.ndarray:
        """
        실제 센서로부터 상태(x)를 읽는다.
        예: [biomass, humidity, temperature, leaf_water]
        """
        # TODO: 실제 센서 연동 코드 작성 (MQTT / Serial / API 등)
        # 여기서는 예시로 난수 사용
        x = np.array([
            random.uniform(0.002, 0.004),  # biomass
            random.uniform(0.001, 0.002),  # humidity
            random.uniform(20, 28),        # temperature
            random.uniform(0.006, 0.009),  # leaf water
        ])
        return x

    # ------------------- 외란 입력 -------------------
    def read_disturbance(self) -> np.ndarray:
        """
        외란(d)을 읽는다.
        예: [radiation, co2, outside_temp, outside_humidity]
        """
        # TODO: 실제 외란 데이터 수집 로직
        d = np.array([
            random.uniform(50, 200),   # radiation
            random.uniform(0.01, 0.03),# co2
            random.uniform(18, 30),    # outside temp
            random.uniform(0.01, 0.02) # outside humidity
        ])
        return d

    # ------------------- 제어기 출력 -------------------
    def apply_control(self, u_opt: np.ndarray):
        """
        MPC가 계산한 제어입력(u_opt)을 실제 장치에 반영한다.
        예: [fan, heater, led]
        """
        # TODO: MQTT / GPIO / Serial 등으로 실제 장비 제어
        fan, heater, led = u_opt
        print(f"[ACTUATOR] Fan={fan:.3f}, Heater={heater:.3f}, LED={led:.3f}")

    # ------------------- 루프 제어 -------------------
    def wait_next_cycle(self):
        """제어 주기까지 대기"""
        time.sleep(self.sample_time)
