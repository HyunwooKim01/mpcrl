# model_real.py
# ---------------------------------------------
# 논문 구조(모델 API)를 유지한 실제환경용 단순 모델
# - df(x, u, d, p)      : 연속시간 동역학
# - euler_step(...)     : 오일러 적분
# - rk4_step(...)       : RK4 적분
# - get_u_min/max()     : 제어 입력 경계
# - get_du_lim()        : 제어 변화율 제한
#
# 인터페이스는 model.py와 동일하며, 내부 식만 실환경 맞게 경량화.

import casadi as cs
import numpy as np


class Model:
    """
    Real-world simplified greenhouse model.
    x = [biomass, humidity, temperature, leaf_water]   (정규화 0~1 가정)
    u = [fan, heater, led]                             (정규화 0~1 가정, heater 0~0.8 권장)
    d = [radiation, co2_ext, outside_temp, outside_humidity]  (정규화 0~1 가정)
    p = (파라미터 벡터, 여기서는 스케일 1.0으로 두되, 인터페이스만 유지)
    """

    # ---- 선택적: 논문과 동일한 형태의 p_true/p_scale 틀만 유지 (내부 계산에 직접 사용하진 않음)
    p_scale = np.ones(4, dtype=float)   # 단순화: 필요시 확장 가능
    p_true = np.ones(4, dtype=float)

    # 제어 입력 경계 (논문 구조: u는 실단위였으나, 실제환경에서는 0~1 정규화)
    @staticmethod
    def get_u_min() -> np.ndarray:
        return np.zeros((3,))

    @staticmethod
    def get_u_max() -> np.ndarray:
        # heater를 과도하게 쓰지 않도록 0.8 제한 (필요시 1.0으로 조정 가능)
        return np.array([1.0, 0.8, 1.0])

    @staticmethod
    def get_du_lim() -> np.ndarray:
        # 한 스텝당 제어 변화율 제한 (논문에도 있는 항목, 실제환경에서 과격한 제어 방지)
        return 0.2 * Model.get_u_max()

    # ---------- 핵심: 연속 동역학 df (논문 API 준수)
    @staticmethod
    def df(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        p: np.ndarray | cs.SX,
    ) -> np.ndarray | cs.SX:
        """
        단순/경량화된 실환경 모델.
        - 논문 모델의 물리적 방향성(부호/상호작용)은 유지
        - 계산량을 줄이기 위해 선형-준선형 근사
        - 외란 d는 radiation, 외기온/습도, 외부 CO2(가용 시)로 구성
        """

        # unpack
        biomass, hum, temp, leafw = x[0], x[1], x[2], x[3]
        fan, heater, led = u[0], u[1], u[2]
        rad, co2_ext, out_temp, out_hum = d[0], d[1], d[2], d[3]

        # --- 생체량: 복사량 및 LED가 증가 요인, 자체 감쇠 포함
        dx1 = 0.001 * (0.7 * rad + 0.3 * led) - 0.0004 * biomass

        # --- 습도: 팬이 감소, 히터가 상승(가열로 상대습도 낮아지나 '수분공급/증발' 항 포함),
        #            외기 습도-내기 습도 차이 확산(외기 습도가 높으면 내부도↑)
        dx2 = -0.05 * fan + 0.02 * heater + 0.03 * (out_hum - hum)

        # --- 온도: 히터↑, 팬↓, 외기-내기 온도차 열교환, 복사 가열 효과
        dx3 = -0.10 * fan + 0.08 * heater + 0.06 * (out_temp - temp) + 0.10 * rad

        # --- 엽면 수분/수분상태: LED/복사로 증산(증발) 증가 경향 + 밤낮 영향(간단 버프),
        #     내부 감쇠 포함
        #     (실제 생리 반응은 더 복잡하지만 계산량을 줄인 근사식)
        dx4 = 0.02 * rad + 0.04 * led - 0.03 * leafw + 0.01 * (rad > 0)

        if isinstance(x, (cs.SX, cs.MX, cs.DM)):
            return cs.vertcat(dx1, dx2, dx3, dx4)
        return np.array([dx1, dx2, dx3, dx4])

    # ---------- 오일러 적분 (논문 model.py에 있는 함수와 동일 API)
    @staticmethod
    def euler_step(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        p: np.ndarray | cs.SX,
        ts: float,
    ) -> np.ndarray | cs.SX:
        return x + ts * Model.df(x, u, d, p)

    # ---------- RK4 적분 (논문 model.py와 동일 API)
    @staticmethod
    def rk4_step(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        p: np.ndarray | cs.SX,
        ts: float,
        steps_per_ts: int = 1,
    ) -> np.ndarray | cs.SX:
        # steps_per_ts 지원(논문 API 호환). 일반적으로 1이면 표준 RK4 한 번.
        def f(x_):
            return Model.df(x_, u, d, p)

        if steps_per_ts == 1:
            k1 = f(x)
            k2 = f(x + (ts / 2) * k1)
            k3 = f(x + (ts / 2) * k2)
            k4 = f(x + ts * k3)
            return x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 여러 substep으로 세분화 적분
        dt = ts / steps_per_ts
        xk = x
        for _ in range(steps_per_ts):
            k1 = f(xk)
            k2 = f(xk + (dt / 2) * k1)
            k3 = f(xk + (dt / 2) * k2)
            k4 = f(xk + dt * k3)
            xk = xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk
