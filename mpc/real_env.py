from typing import Dict
import numpy as np

class RealSensorEnv:
    """실제 센서 데이터를 기반으로 하는 MPC 환경"""
    def __init__(self, state_keys=["temp","hum","co2","light"]):
        self.state_keys = state_keys
        self.nx = len(state_keys)  # 상태 차원
        self.nu = 4  # 액추에이터 차원: heater, humidifier, co2_valve, led
        self.current_state = np.zeros(self.nx)
        self.previous_action = np.zeros(self.nu)

    def set_state(self, state_dict: Dict[str, float]):
        """MQTT로 받은 센서 상태를 Env에 적용"""
        for i,k in enumerate(self.state_keys):
            if k in state_dict:
                self.current_state[i] = float(state_dict[k])

    def step(self, action: np.ndarray):
        """실제 환경에서는 상태는 센서로 받으므로 step은 단순 저장"""
        self.previous_action = action
        # 여기서는 MPC 계산용 state와 cost 계산에 활용할 수 있음
        return self.current_state.copy(), 0.0  # reward는 시뮬레이션에서만 의미

    def get_state_vector(self):
        """MPC에 입력할 상태 벡터 반환"""
        return self.current_state.copy()
