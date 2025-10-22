# default_real.py
# -----------------
# 실제환경에서 LearningMpcReal이 사용하는 최소 설정값 모음
# 강화학습, ExperienceReplay, Exploration 등의 항목 제거됨
# 
# 목적: 실시간 제어 환경 (센서 입력 기반)에서 MPC를 안정적으로 테스트하기 위한 파라미터 구성
# -------------------------------------------------------------
# 실제환경용 MPC 설정 (작물 프로필 JSON 기반 자동화)
# -------------------------------------------------------------
from typing import Any
import numpy as np
import json
import os

class DefaultReal:
    """실제 환경용 MPC 설정 (crop_profile.json 자동 적용)."""
    
    def __init__(self, crop_name: str = "lettuce"):
        """생성 시 자동으로 crop profile을 불러와 learnable_pars_init 설정"""
        self.crop_name = crop_name
        self.learnable_pars_init = self.make(crop_name)

    # ---------------------------
    # 기본 MPC 설정
    # ---------------------------
    discount_factor = 0.99
    p_learn = []
    p_perturb = []
    fixed_pars: dict[str, Any] = {}

    # ---------------------------
    # 센서 정규화 기준 (real_env.py와 동일)
    # ---------------------------
    sensor_norm_ranges = {
    "biomass": (0.0, 0.005),       # 그대로
    "humidity": (40.0, 80.0),      # 실제 온실 범위로 좁힘
    "temperature": (20.0, 30.0),   # 온실 정상 운영 범위
    "leaf_water": (0.006, 0.009),  # 그대로
    }

    # ---------------------------
    # JSON 로드 함수
    # ---------------------------
    @staticmethod
    def load_crop_profile(crop_name: str) -> dict:
        base_dir = os.path.join(os.path.dirname(__file__), "../../crops")
        file_path = os.path.abspath(os.path.join(base_dir, f"{crop_name}_profile.json"))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Crop profile not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    # ---------------------------
    # 정규화 함수
    # ---------------------------
    @classmethod
    def normalize(cls, value, min_val, max_val, key):
        """sensor normalization"""
        return (value - min_val) / (max_val - min_val)

    # ---------------------------
    # y_fin, c_y, c_u 생성
    # ---------------------------
    @classmethod
    def make(cls, crop_name: str = "lettuce"):
        profile = cls.load_crop_profile(crop_name)
        targets = profile["targets"]
        priorities = profile["priority"]
        control_w = profile["control_weights"]
        sr = cls.sensor_norm_ranges

        # 목표값 정규화
        y_fin = np.array([
            targets["biomass"],                           # 그대로 사용
            cls.normalize(targets["humidity"], *sr["humidity"], "humidity"),
            cls.normalize(targets["temperature"], *sr["temperature"], "temperature"),
            targets["leaf_water"]                         # 그대로 사용
        ])

        # 상태 추종 가중치(Q)
        c_y = np.array([
            priorities["biomass"],
            priorities["humidity"],
            priorities["temperature"],
            priorities["leaf_water"]
        ])

        # 제어 가중치(R)
        c_u = np.array([
            control_w["fan"],
            control_w["heater"],
            control_w["led"]
        ])

        return {
            "V0": np.zeros((1,)),
            "c_dy": 10 * np.ones((1,)),
            "w": 1e3 * np.ones((4,)),
            "olb": np.zeros((4,)),
            "oub": np.zeros((4,)),

            # 🌱 정규화된 목표값 (4x1 column vector)
            "y_fin": np.array([y_fin[2]]),   # 온도 하나만 (shape: (1,))

            # 상태 가중치 (Q)
            "c_y": np.array([1500.0]), 

            # 제어 입력 가중치 (R)
            #"c_u": c_u,
            "c_u": np.array([0.05, 0.02, 0.01])  # fan, heater, led 민감도 ↑
        }


