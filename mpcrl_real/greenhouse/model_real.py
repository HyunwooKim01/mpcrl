# greenhouse/model_real.py
# -----------------------------------------------------
# 논문 기반 실환경용 Greenhouse 동적 모델 (MPC 예측 가능)
# -----------------------------------------------------
# 특징:
# - 상태 x = [biomass, humidity, temperature, leaf_water]
# - 제어 u = [fan, heater, led]
# - 외란 d = [radiation, co2_ext, outside_temp, outside_humidity]
# -----------------------------------------------------
#  • 논문식 열/습도 에너지 수지 방정식을 단순화하여 포함
#  • 현재 상태 x 가 다음 상태에 직접 영향 → MPC가 “예측” 가능
#  • CasADi 및 Numpy 양쪽 호환
# -----------------------------------------------------

import casadi as cs
import numpy as np

class Model:
    """논문식 실환경용 동적 Greenhouse 모델"""

    n_params = 4
    p_scale = np.ones(4)
    p_true = np.ones(4)

    # ──────────────────────────────────────────────
    # 제어 입력 경계 및 변화율 제한
    # ──────────────────────────────────────────────
    @staticmethod
    def get_u_min():
        return np.zeros((3,))

    @staticmethod
    def get_u_max():
        return np.array([1.0, 0.8, 1.0])  # fan, heater, led

    @staticmethod
    def get_du_lim():
        return 0.2 * Model.get_u_max()

    # ──────────────────────────────────────────────
    # 핵심: 동적 모델 df(x,u,d,p)
    # 논문 구조를 단순화한 형태로 구현
    # ──────────────────────────────────────────────
    @staticmethod
    def df(x, u, d, p):
        """
        x = [biomass, humidity, temperature, leaf_water]
        u = [fan, heater, led]
        d = [radiation, co2_ext, outside_temp, outside_humidity]
        """

        biomass, hum, temp, leafw = x[0], x[1], x[2], x[3]
        fan, heater, led = u[0], u[1], u[2]
        rad, co2, out_temp, out_hum = d[0], d[1], d[2], d[3]

        # ── 논문식 greenhouse 근사 모델 ──
        # (물리 기반 동적항 포함)
        # 온도 변화율
        dx_temp = (
            0.10 * (out_temp - temp)     # 외기-내기 열교환
            - 0.12 * fan                 # 팬 냉각 효과
            + 0.20 * heater              # 히터 가열 효과
            + 0.08 * rad                 # 복사열 영향
        )

        # 습도 변화율
        dx_hum = (
            0.05 * (out_hum - hum)       # 외기 습도 확산
            - 0.06 * fan                 # 팬에 의한 건조
            + 0.04 * heater              # 온도 상승으로 인한 증발
            - 0.03 * (temp - 0.6)        # 온도에 의한 추가 증발항
        )

        # 엽면 수분
        dx_leaf = (
            0.001 * (hum - 0.6)          # 습도 높을수록 유지
            - 0.0008 * (temp - 0.6)      # 온도 높을수록 증발
            + 0.0006 * rad               # 복사로 인한 증발
            - 0.0005 * leafw             # 자연 감소항
        )

        # 생체량 (성장)
        dx_biomass = (
            0.0005 * led                 # 광합성 (조도 영향)
            + 0.0003 * rad               # 복사 영향
            - 0.0002 * abs(temp - 0.55)  # 온도 스트레스
            - 0.0001 * (1 - hum)         # 습도 부족 스트레스
        )

        if isinstance(x, (cs.SX, cs.MX, cs.DM)):
            return cs.vertcat(dx_biomass, dx_hum, dx_temp, dx_leaf)
        return np.array([dx_biomass, dx_hum, dx_temp, dx_leaf])

    # ──────────────────────────────────────────────
    # 수치 적분 (Euler / RK4)
    # ──────────────────────────────────────────────
    @staticmethod
    def euler_step(x, u, d, p, ts):
        return x + ts * Model.df(x, u, d, p)

    @staticmethod
    def rk4_step(x, u, d, p, ts, steps_per_ts=1):
        def f(x_): return Model.df(x_, u, d, p)
        dt = ts / steps_per_ts
        xk = x
        for _ in range(steps_per_ts):
            k1 = f(xk)
            k2 = f(xk + 0.5 * dt * k1)
            k3 = f(xk + 0.5 * dt * k2)
            k4 = f(xk + dt * k3)
            xk = xk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return xk

    # ──────────────────────────────────────────────
    # 출력 관련 (센서 연동용)
    # ──────────────────────────────────────────────
    @staticmethod
    def output(x, p):
        return x

    @staticmethod
    def get_output_range():
        y_min = np.array([0.0, 0.3, 0.3, 0.005])
        y_max = np.array([0.01, 0.9, 0.9, 0.009])
        return y_min, y_max

    @staticmethod
    def get_true_parameters():
        return np.array([0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def get_output_min(d=None):
        y_min, _ = Model.get_output_range()
        return y_min

    @staticmethod
    def get_output_max(d=None):
        _, y_max = Model.get_output_range()
        return y_max
