import os
import json, time
import numpy as np
from real_env import RealEnvironment
from sims.configs.default_real import DefaultReal
from learning_real import LearningMpcReal
from greenhouse.model_real import Model


# ────────────────────────────────────────────────
# 🥬 1️⃣ 작물 프로필 로드 (안전/견고 버전)
# ────────────────────────────────────────────────
def load_crop_profile(crop_name: str = "lettuce"):
    """JSON에서 작물 프로필 로드 (BOM/경로/키 불일치 방어)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "crops", f"{crop_name.lower()}_profile.json")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
        text = raw.strip()
        profile = json.loads(text)  # 앞뒤 공백 제거 후 파싱

        # 키 이름 호환: crop / crop_name
        crop_label = profile.get("crop") or profile.get("crop_name") or crop_name
        desc = profile.get("description", "-")
        print(f"🌿 Loaded crop profile: {crop_label} ({desc}) @ {path}")
        return profile
    except FileNotFoundError:
        print(f"⚠️ {path} 파일 없음 → DefaultReal 기본값 사용")
        return None
    except json.JSONDecodeError as e:
        # 디버깅 힌트 제공
        snippet_head = text[:200].replace("\n", "\\n")
        snippet_tail = text[-200:].replace("\n", "\\n")
        print("❌ JSON 파싱 오류:", e)
        print(f"   ▸ 파일 경로: {path}")
        print(f"   ▸ 앞쪽 스니펫: {snippet_head}")
        print(f"   ▸ 뒤쪽 스니펫: {snippet_tail}")
        print("   ▸ 점검: 마지막 항목 뒤 쉼표 제거, 주석(//, /* */) 제거, JSON 객체 1개만 존재하는지 확인")
        return None


# ────────────────────────────────────────────────
# 🚀 2️⃣ 메인 루프
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Real-world MPC controller starting...")

    # (1) 환경 초기화
    env = RealEnvironment(sample_time=5.0)
    crop_name = "lettuce"   # 🍅 tomato / 🥒 cucumber / 🍓 strawberry

    # (2) 작물 프로필 로드 + MPC 파라미터 자동 세팅
    crop_profile = load_crop_profile(crop_name)
    crop_config = DefaultReal(crop_name)   # 내부에서 learnable_pars_init 자동 생성

    # (3) MPC 컨트롤러 초기화
    mpc = LearningMpcReal(test=crop_config)
    print("✅ MPC controller initialized.")

    # (4) 제어 루프
    while True:
        try:
            # 내부/외부 환경 읽기
            x_current = env.read_sensors()
            d_current = env.read_disturbance()

            # 출력 시 프로필 유무 안전 처리
            crop_label = (crop_profile or {}).get("crop") \
                         or (crop_profile or {}).get("crop_name") \
                         or crop_name

            # 상태 출력
            print("\n📡 [INPUT SUMMARY]")
            print(f"  내부환경 x : {x_current.round(3)}")
            print(f"  외부환경 d : {d_current.round(3)}")
            print(f"  작물정보   : {crop_label}")
            print("------------------------------------------------------------")

            # MPC 계산
            u_opt, status = mpc.compute_control(x_current, d_current)
            print(f"[MPC] status={status}, u_opt={u_opt.round(3)}")

            # 제어값 MQTT 전송
            env.apply_control(u_opt)

            # 루프 주기 대기
            env.wait_next_cycle()

        except KeyboardInterrupt:
            print("🛑 MPC control loop stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(3)
