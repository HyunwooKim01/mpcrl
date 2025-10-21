"""
model_real.py
-------------
실제 환경에서 사용하는 단순 greenhouse 모델.
- Fan, Heater, LED의 제어 효과를 선형 근사.
- 센서값(x)과 외란(d)을 입력받아 상태 변화량(dx)을 계산.
- LearningMpcReal에서 사용됨.
"""

import numpy as np
import casadi as cs


class RealModel:
    """Real-world simplified greenhouse dynamics model."""

    # ------------------ 제어 입력 경계 ------------------
    @staticmethod
    def get_u_min() -> np.ndarray:
        """각 제어 입력의 최소값 (Fan, Heater, LED)"""
        return np.array([0.0, 0.0, 0.0])  # OFF 상태

    @staticmethod
    def get_u_max() -> np.ndarray:
        """각 제어 입력의 최대값 (Fan, Heater, LED)"""
        return np.array([1.0, 0.8, 1.0])  # Heater는 80% 제한

    @staticmethod
    def get_du_lim() -> np.ndarray:
        """한 주기당 제어 변화율 제한 (Δu)"""
        return np.array([0.2, 0.2, 0.2])  # 너무 급격한 제어 방지

    # ------------------ 시스템 동역학 ------------------
    @staticmethod
    def dynamics(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
    ) -> np.ndarray | cs.SX:
        """
        단순 greenhouse 동역학 (정규화 스케일 기준)
        x = [biomass, humidity, temperature, leaf_water]
        u = [fan, heater, led]
        d = [radiation, co2, outside_temp, outside_humidity]
        """
        dx1 = 0.001 * d[0] - 0.0005 * x[0]           # biomass: 광량 영향
        dx2 = -0.05 * u[0] + 0.02 * u[1] + 0.02 * (d[3] - x[1])  # humidity
        dx3 = -0.10 * u[0] + 0.08 * u[1] + 0.05 * (d[2] - x[2])  # temperature
        dx4 =  0.05 * u[2] - 0.02 * x[3] + 0.01 * (d[0] - 0.5)   # leaf_water

        if isinstance(x, (cs.SX, cs.MX, cs.DM)):
            return cs.vertcat(dx1, dx2, dx3, dx4)
        return np.array([dx1, dx2, dx3, dx4])

    # ------------------ RK4 적분기 ------------------
    @staticmethod
    def step_rk4(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        ts: float,
    ) -> np.ndarray | cs.SX:
        """Runge-Kutta 4차 적분 (MPC 예측모델용)"""
        f = RealModel.dynamics
        k1 = f(x, u, d)
        k2 = f(x + ts / 2 * k1, u, d)
        k3 = f(x + ts / 2 * k2, u, d)
        k4 = f(x + ts * k3, u, d)
        return x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
