"""
main.py
--------
실제환경용 MPC 루프 (InputManager 기능 통합)
 - 내부환경, 외부환경, 작물정보를 자동 수집
 - 작물 JSON 불러와 MPC 파라미터 자동 반영
"""

import json, time
import numpy as np
from real_env import RealEnvironment
from sims.configs.default_real import DefaultReal        # ✅ 경로 수정됨
from learning_real import LearningMpcReal
from greenhouse.model_real import Model                  # ✅ greenhouse 경로에 맞춤

# ────────────────────────────────────────────────
# 🥬 1️⃣ 작물 프로필 불러오기
# ────────────────────────────────────────────────
def load_crop_profile(crop_name: str = "lettuce"):
    path = f"crops/{crop_name.lower()}_profile.json"
    try:
        with open(path, "r") as f:
            profile = json.load(f)
            print(f"🌿 Loaded crop profile: {profile['name']} ({profile['stage']})")
            return profile
    except FileNotFoundError:
        print(f"⚠️ {path} 파일 없음 → DefaultReal 기본값 사용")
        return {}

def make_crop_config(profile: dict) -> DefaultReal:
    cfg = DefaultReal()
    weights = profile.get("weights", {})
    if weights:
        cfg.learnable_pars_init["c_y"] = np.array([weights.get("c_y", 10)])
        cfg.learnable_pars_init["c_dy"] = np.array([weights.get("c_dy", 10)])
        cfg.learnable_pars_init["c_u"] = np.array(weights.get("c_u", [1, 0.5, 0.5]))
        cfg.learnable_pars_init["w"] = np.array(weights.get("w", [1000, 1000, 1000, 1000]))
    cfg.learnable_pars_init["y_fin"] = np.array([profile.get("T_ref_day", 25.0)])
    return cfg

# ────────────────────────────────────────────────
# 🚀 2️⃣ 메인 루프 시작
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Real-world MPC controller starting...")

    # (1) 환경 및 작물 초기화
    env = RealEnvironment(sample_time=5.0)
    crop_name = "lettuce"  # 🔄 바꾸면 자동 적용 ("tomato", "strawberry" 등)
    crop_profile = load_crop_profile(crop_name)
    crop_config = make_crop_config(crop_profile)

    # (2) MPC 초기화 (작물 프로필 반영)
    mpc = LearningMpcReal(test=crop_config)
    print("✅ MPC controller initialized.")

    # (3) 제어 루프
    while True:
        try:
            # 내부/외부 환경 읽기
            x_current = env.read_sensors()
            d_current = env.read_disturbance()

            # 입력 상태 요약 출력
            print("\n📡 [INPUT SUMMARY]")
            print(f"  내부환경 x : {x_current.round(3)}")
            print(f"  외부환경 d : {d_current.round(3)}")
            print(f"  작물정보   : {crop_profile.get('name','unknown')} ({crop_profile.get('stage','-')})")
            print("-" * 60)

            # MPC 계산
            u_opt, status = mpc.compute_control(x_current, d_current)
            print(f"[MPC] status={status}, u_opt={u_opt.round(3)}")

            # 제어값 적용 (ESP로 전송)
            env.apply_control(u_opt)

            # 다음 주기 대기
            env.wait_next_cycle()

        except KeyboardInterrupt:
            print("🛑 MPC control loop stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(3)
