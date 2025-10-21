# default_real.py
# -----------------
# 실제환경에서 LearningMpcReal이 사용하는 최소 설정값 모음
# 강화학습, ExperienceReplay, Exploration 등의 항목 제거됨
# 
# 목적: 실시간 제어 환경 (센서 입력 기반)에서 MPC를 안정적으로 테스트하기 위한 파라미터 구성

from typing import Any
import numpy as np
from greenhouse.model_real import RealModel


class DefaultReal:
    """실제 환경용 MPC 테스트 설정 (no RL, no replay buffer)."""

    # ---------------------------
    # MPC 파라미터 설정
    # ---------------------------
    discount_factor = 0.99  # 시간 할인율 (γ)

    # 모델 관련 파라미터 (고정)
    p_learn = []  # 실제환경에서는 학습 안 함
    p_perturb = []  # 섭동 없음 (Model.get_true_parameters() 사용)
    fixed_pars: dict[str, Any] = {}  # 고정된 파라미터 (없음)

    # ---------------------------
    # 비용함수 가중치 설정
    # ---------------------------
    learnable_pars_init = {
        # 초기비용 (보통 0)
        "V0": np.zeros((1,)),

        # 출력 변화율(cost_dy): 작을수록 반응 빠름
        "c_dy": 10 * np.ones((1,)),

        # 제약 위반 패널티(slack weight): 제약조건 위반 방지용
        "w": 1e3 * np.ones((4,)),

        # 출력 제약 보정(lower/upper bounds)
        "olb": np.zeros((4,)),
        "oub": np.zeros((4,)),

        # 최종 목표 출력값 (예: 내부 온도 26°C)
        "y_fin": 26 * np.ones((1,)),

        # 최종 상태 추종 가중치: 높을수록 목표 도달 강제
        "c_y": 10 * np.ones((1,)),

        # 제어 입력비용: 낮을수록 제어기 적극적으로 작동
        # 순서: [fan, heater, led]
        "c_u": np.array([1, 0.5, 0.5]),
    }

    # ---------------------------
    # 파라미터 경계 (학습 모드일 때만 사용)
    # ---------------------------
    learn_bounds = {
        "c_dy": [0, np.inf],
        "w": [0, np.inf],
        "olb": [-0.2, 0.2],
        "oub": [-0.2, 0.2],
        "y_fin": [0, np.inf],
        "c_y": [0, np.inf],
        "c_u": [0, np.inf],
    }
