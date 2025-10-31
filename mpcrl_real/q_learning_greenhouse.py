# q_learning_real.py
# ───────────────────────────────────────────────────────────────
# Real-sensor Q-learning for RL-MPC (no simulator).
#   - Sensor/Disturbance via MQTT (real_env.py)
#   - CasADi MPC policy (learning_real_detail.py: LearningMpcCasADi)
#   - Reward: env.compute_reward (paper Eqs. (18)–(21) structure)
#   - θ = {Q, R, S, alpha_growth} online update (paper Eq. (22) style)
#   - Auto-save θ every 6 hours to trained_theta.pkl and hot-reload
# ───────────────────────────────────────────────────────────────

from __future__ import annotations
import os, sys, time, json, math, pickle, signal, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# ── Project modules
from real_env import RealEnvironment
try:
    # 너가 올린 학습용 MPC 클래스 (CasADi 기반)
    from learning_real_detail import LearningMpcCasADi
except Exception as e:
    print(f"[FATAL] learning_real_detail import 실패: {e}")
    sys.exit(1)

# ───────────────────────────────────────────────────────────────
# 설정
# ───────────────────────────────────────────────────────────────
DEFAULT_THETA_PATH = "trained_theta.pkl"

@dataclass
class HyperParams:
    gamma: float = 0.99          # discount (TD형 근사에서 사용)
    lr_q: float = 1e-3           # Q 가중 업데이트 학습률
    lr_r: float = 1e-3           # R 가중 업데이트 학습률
    lr_s: float = 2e-3           # S 가중 업데이트 학습률 (제약 위반에 더 민감)
    lr_alpha: float = 2e-3       # 성장 가중 업데이트 학습률
    clip_step: float = 0.05      # 가중 한 스텝 변화율 클립(±5%)
    save_interval_s: int = 6*3600  # 6시간마다 저장
    horizon_N: int = 24          # MPC horizon (≈ 6h if 15min, 또는 실계측 주기에 맞춤)
    warmup_steps: int = 5        # 초기 과도 스텝 (du 큰 영향 방지)
    max_q: float = 1e3           # 안정용 상한
    max_r: float = 1e2
    max_s: float = 1e3
    alpha_bounds: Tuple[float,float] = (0.1, 10.0)

HP = HyperParams()

# ───────────────────────────────────────────────────────────────
# θ 로드/저장 & MPC 반영
# ───────────────────────────────────────────────────────────────
def load_theta(path: str) -> Dict:
    if not os.path.exists(path):
        # 기본값: 온도/습도 추적은 높게, Δu/에너지는 보통, 제약 위반은 강하게
        return {
            "Q": [2.0, 2.0, 0.0, 0.0],      # [temp, hum, co2, light] 추적 가중(필요시 확장)
            "R": [0.05, 0.05, 0.02],        # [fan, heater, led] 제어 가중
            "S": [5.0, 5.0],                # [temp_violation, hum_violation]
            "alpha_growth": 1.0,            # 성장 보상 가중
        }
    with open(path, "rb") as f:
        return pickle.load(f)

def save_theta(theta: Dict, path: str = DEFAULT_THETA_PATH):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(theta, f)
    os.replace(tmp, path)
    print(f"💾 θ saved → {path}")

def apply_theta_to_mpc(mpc: LearningMpcCasADi, theta: Dict):
    """mpc 내부의 가중치 행렬/스칼라를 θ로 갱신. (학습 클래스 인터페이스에 맞춰 적용)"""
    try:
        # Q, R, S가 np.diag로 정의되어 있다는 가정(너의 학습 클래스가 이 형태)
        Qd = np.array(theta.get("Q", []), dtype=float)
        Rd = np.array(theta.get("R", []), dtype=float)
        Sd = np.array(theta.get("S", []), dtype=float)

        if Qd.size > 0:
            mpc.Q = np.diag(Qd)
        if Rd.size > 0:
            mpc.R = np.diag(Rd)
        if Sd.size > 0:
            # temp/hum 위반 슬랙 가중치만 반영 (필요시 확장)
            mpc.S = np.diag(Sd)

        if "alpha_growth" in theta:
            mpc.alpha_growth = float(theta["alpha_growth"])

        # 혹시 클래스에 제공되는 편의함수가 있다면 사용
        if hasattr(mpc, "on_theta_updated"):
            mpc.on_theta_updated()

    except Exception as e:
        print(f"[WARN] apply_theta_to_mpc 실패: {e}")

# ───────────────────────────────────────────────────────────────
# 보조: 지표 추정(실측 기반)
# ───────────────────────────────────────────────────────────────
def estimate_terms(x: np.ndarray,
                   x_ref_ranges: Dict,
                   u: np.ndarray,
                   u_prev: np.ndarray) -> Dict[str, float]:
    """
    논문 식 (18)–(21) 항들을 현실적으로 근사해서 스칼라 지표로 요약.
    - 추적오차: 온도/습도 중심값 기준 제곱오차
    - 제약위반: 범위 넘어선 총량(제곱 누적)
    - 에너지: fan/heater 제곱합
    - Δu: 제어 변화율 제곱합
    - 성장: T/H/L의 간단한 가우시안/포화 근사(= real_env.compute_reward와 일치하게)
    """
    temp, hum, co2, light = [float(v) for v in x]
    Tmin, Tmax = x_ref_ranges.get("target_temp", [18.0, 22.0])
    Hmin, Hmax = x_ref_ranges.get("target_humidity", [50.0, 70.0])
    Tref = 0.5*(Tmin+Tmax); Href = 0.5*(Hmin+Hmax)

    err_T = (temp - Tref)**2
    err_H = (hum  - Href)**2
    viol_T = max(0.0, Tmin-temp) + max(0.0, temp-Tmax)
    viol_H = max(0.0, Hmin-hum ) + max(0.0, hum -Hmax)
    viol = viol_T**2 + viol_H**2

    fan, heater, led = [float(np.clip(v,0,1)) for v in u]
    energy = fan**2 + heater**2
    du = u - u_prev
    du2 = float(np.sum(du*du))

    G_temp = math.exp(-0.5 * ((temp - 25.0) / 2.5) ** 2)
    G_hum  = math.exp(-0.5 * ((hum  - 60.0) / 8.0)  ** 2)
    G_light = math.tanh(light / 500.0)
    growth = G_temp * G_hum * G_light

    return dict(err_T=err_T, err_H=err_H, viol=viol, energy=energy, du2=du2, growth=growth)

# ───────────────────────────────────────────────────────────────
# 파라미터 업데이트 규칙 (Q-learning style heuristic)
# ───────────────────────────────────────────────────────────────
def update_theta(theta: Dict, terms: Dict[str,float]):
    """
    논문 (22)의 파라미터화된 MPC 비용을 현실적으로 근사:
      - Q: 추적오차(err_T, err_H)가 크면 증가, 작으면 완만히 감소
      - S: 제약위반(viol)이 크면 강하게 증가
      - R: Δu/에너지 크면 증가
      - alpha_growth: growth가 클수록 조금 증가(성장에 보상), 위반 크면 감소(안전 우선)
    전체는 작은 학습률과 변화율 클립으로 안정화.
    """
    Q = np.array(theta["Q"], dtype=float)       # [temp, hum, (co2), (light)]
    R = np.array(theta["R"], dtype=float)       # [fan, heater, led]
    S = np.array(theta["S"], dtype=float)       # [temp_slack, hum_slack]
    alpha = float(theta["alpha_growth"])

    # — 스케일링 (안정화를 위한 작은 비율)
    dq_T =  (+HP.lr_q * terms["err_T"]) - (0.25*HP.lr_q * max(0.0, 0.02-terms["err_T"]))
    dq_H =  (+HP.lr_q * terms["err_H"]) - (0.25*HP.lr_q * max(0.0, 0.02-terms["err_H"]))
    dS   =  (+HP.lr_s * terms["viol"])
    dR   =  (+HP.lr_r * (0.6*terms["du2"] + 0.4*terms["energy"]))

    # — 성장/안전 트레이드오프
    dalpha =  (+HP.lr_alpha * (terms["growth"] - 0.2)) - (HP.lr_alpha * 0.1 * (terms["viol"]>0))

    # — 적용 (클립 & 경계)
    def step_clip(v, dv, vmax):
        delta = np.clip(dv, -HP.clip_step*max(1.0,abs(v)), HP.clip_step*max(1.0,abs(v)))
        return float(np.clip(v + delta, 0.0, vmax))

    # Q: temp→0, hum→1 인덱스 가정
    if Q.size >= 2:
        Q[0] = step_clip(Q[0], dq_T, HP.max_q)
        Q[1] = step_clip(Q[1], dq_H, HP.max_q)

    # S: temp/hum slack
    if S.size >= 2:
        S[0] = step_clip(S[0], dS, HP.max_s)
        S[1] = step_clip(S[1], dS, HP.max_s)

    # R: 모든 입력에 동일 증분(단순화)
    for i in range(R.size):
        R[i] = step_clip(R[i], dR, HP.max_r)

    alpha = float(np.clip(alpha + dalpha, HP.alpha_bounds[0], HP.alpha_bounds[1]))

    theta["Q"] = Q.tolist()
    theta["R"] = R.tolist()
    theta["S"] = S.tolist()
    theta["alpha_growth"] = alpha

# ───────────────────────────────────────────────────────────────
# 메인 루프
# ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker_host", type=str, default="172.27.148.207")
    ap.add_argument("--broker_port", type=int, default=1883)
    ap.add_argument("--farm_id", type=str, default="farmA")
    ap.add_argument("--esp_id", type=str, default="esp1")
    ap.add_argument("--sample_time", type=float, default=5.0)
    ap.add_argument("--theta_path", type=str, default=DEFAULT_THETA_PATH)
    ap.add_argument("--horizon", type=int, default=HP.horizon_N)
    args = ap.parse_args()

    # 1) Real env (MQTT)
    env = RealEnvironment(
        sample_time=args.sample_time,
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        farm_id=args.farm_id,
        esp_id=args.esp_id,
    )

    # 2) MPC (CasADi)
    try:
        mpc = LearningMpcCasADi(ts=env.sample_time, N=args.horizon)
    except TypeError:
        # 시그니처가 다르면 합리적 기본값 사용
        mpc = LearningMpcCasADi()

    # 3) θ 로드 & MPC 적용
    theta = load_theta(args.theta_path)
    apply_theta_to_mpc(mpc, theta)
    # 3) θ 로드 & MPC 적용
    theta = load_theta(args.theta_path)
    apply_theta_to_mpc(mpc, theta)
    # numpy 배열이 섞여 있을 수 있으므로 안전하게 출력
    theta_safe = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in theta.items()}
    print("🔧 θ loaded & applied:", json.dumps(theta_safe, indent=2, ensure_ascii=False))


    # 4) 루프 준비
    u_prev = np.zeros(3, dtype=float)
    last_save = time.time()
    step = 0

    # 안전한 종료
    stop_flag = {"stop": False}
    def handle_sig(sig, frm):
        stop_flag["stop"] = True
        print("\n🛑 Stopping... (saving θ)")
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # 참조 범위(보상에 사용)
    crop = env.crop  # real_env 내부에서 로드됨

    print("🚀 Start real Q-learning loop (no simulator)")
    while not stop_flag["stop"]:
        try:
            # (1) 현재 상태
            x, d = env.read_sensors()
            s = np.concatenate([x, d])

            # (2) MPC policy (u_opt)
            u_opt, J_mpc = None, None
            try:
                # 네가 만든 클래스에 맞게 policy 반환값 사용
                # (u_opt, J) 또는 u_opt 만 반환 가능성 둘 다 대응
                out = mpc.policy(s)
                if isinstance(out, tuple) and len(out) >= 1:
                    u_opt = np.array(out[0], dtype=float).reshape(-1)
                    if len(out) >= 2:
                        J_mpc = float(out[1])
                else:
                    u_opt = np.array(out, dtype=float).reshape(-1)
            except Exception as e:
                print(f"[WARN] MPC policy 실패: {e}")
                u_opt = np.zeros(3, dtype=float)

            # (3) 액추에이터 전송
            env.send_actuators(u_opt)

            # (4) 다음 상태 관측
            time.sleep(env.sample_time)
            x_next, d_next = env.read_sensors()

            # (5) 보상
            r = env.compute_reward(x_next, u_opt, u_prev=u_prev, J_mpc=J_mpc)

            # (6) θ 업데이트 (워밍업 후)
            if step >= HP.warmup_steps:
                terms = estimate_terms(x_next, crop, u_opt, u_prev)
                update_theta(theta, terms)
                apply_theta_to_mpc(mpc, theta)

            u_prev = u_opt.copy()
            step += 1

            # (7) 주기 저장 & 핫 리로드(외부에서 파일이 갱신된 경우 대비)
            now = time.time()
            if (now - last_save) >= HP.save_interval_s:
                save_theta(theta, args.theta_path)
                # 외부 갱신 우선시할 경우: 새로 로드하여 합치고자 하면 주석 해제
                # theta_ext = load_theta(args.theta_path)
                # theta = theta_ext
                last_save = now
                print("🧠 θ periodic update done.")

        except Exception as e:
            print(f"[LOOP WARN] {e}")
            time.sleep(min(10.0, env.sample_time*2))

    # 종료 시 저장
    save_theta(theta, args.theta_path)
    print("✅ Exit cleanly.")

if __name__ == "__main__":
    main()
