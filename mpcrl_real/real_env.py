"""
real_env.py
------------
실제 센서 데이터를 모델 스케일(0~1)로 정규화하여 MPC로 전달하는 환경 클래스.
현재는 랜덤 데이터로 시뮬레이션하지만,
실제 센서 연동 시에도 동일하게 값을 정규화만 거쳐 넘기면 된다.
"""

import time
import random
import numpy as np


class RealEnvironment:
    def __init__(self, sample_time: float = 5.0):
        """
        sample_time : 제어 루프 주기 (초 단위)
        """
        self.sample_time = sample_time

    # ------------------- 센서 입력 -------------------
    def read_sensors(self) -> np.ndarray:
        """
        실제 센서에서 읽은 데이터를 모델 스케일(0~1)로 정규화.
        x = [biomass, humidity, temperature, leaf_water]
        """
        # 실제 환경 센서값 (예시, 추후 실제값으로 대체)
        temp_c = random.uniform(20.0, 28.0)     # °C
        hum_per = random.uniform(40.0, 80.0)    # %
        biomass = random.uniform(0.002, 0.004)  # kg 단위 추정
        leaf_water = random.uniform(0.006, 0.009)

        # -------- 정규화 (Model expects 0~1 range) --------
        # 15~35°C → 0~1
        temp_norm = (temp_c - 15.0) / 20.0
        # 0~100% → 0~1
        hum_norm = hum_per / 100.0

        x_norm = np.array([
            biomass,      # 이미 작은 단위 (0~0.01 근처)
            hum_norm,     # 정규화된 습도
            temp_norm,    # 정규화된 온도
            leaf_water    # 이미 작은 단위
        ])

        return x_norm

    # ------------------- 외란 입력 -------------------
    def read_disturbance(self) -> np.ndarray:
        """
        외부 환경 요인 (복사량, CO₂, 외기온도, 외기습도)을 정규화하여 반환.
        d = [radiation, co2, outside_temp, outside_humidity]
        """
        # 실제 센서값 예시
        rad_w = random.uniform(50.0, 200.0)     # W/m²
        co2_ppm = random.uniform(400.0, 2000.0) # ppm
        out_temp_c = random.uniform(15.0, 30.0) # °C
        out_hum_per = random.uniform(40.0, 90.0)# %

        # -------- 정규화 (Model expects 0~1 range) --------
        # 복사량: 0~200 W/m² → 0~1
        rad_norm = rad_w / 200.0
        # CO₂: 400~2000 ppm → 0~1
        co2_norm = (co2_ppm - 400.0) / 1600.0
        # 외기온: 15~35°C → 0~1
        out_temp_norm = (out_temp_c - 15.0) / 20.0
        # 외기습도: 0~100% → 0~1
        out_hum_norm = out_hum_per / 100.0

        d_norm = np.array([
            rad_norm,
            co2_norm,
            out_temp_norm,
            out_hum_norm
        ])

        return d_norm

    # ------------------- 제어 출력 -------------------
    def apply_control(self, u_opt: np.ndarray):
        """
        MPC가 계산한 제어 입력을 출력.
        실제 환경에서는 이 값을 PWM, 릴레이, MQTT 등으로 전달 가능.
        """
        fan, heater, led = u_opt
        print(f"[ACTUATOR] Fan={fan:.3f}, Heater={heater:.3f}, LED={led:.3f}")

    # ------------------- 루프 주기 대기 -------------------
    def wait_next_cycle(self):
        """제어 주기 대기"""
        time.sleep(self.sample_time)
