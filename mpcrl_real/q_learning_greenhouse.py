# q_learning_greenhouse.py
# ────────────────────────────────────────────────
# RL Learner (MQTT-based) — 학습 전용, 제어 수행 X
#   • Subscribes: sensor, disturbance, mpc/u_opt
#   • Computes reward via real_env.compute_reward()
#   • Updates θ = {Q,R,S,α} (Eq.22-style)
#   • Saves to trained_theta.pkl every 6 hours
# ────────────────────────────────────────────────

from __future__ import annotations
import os, sys, time, json, math, pickle, signal, threading
import numpy as np
import paho.mqtt.client as mqtt
from dataclasses import dataclass
from typing import Dict, Tuple
from real_env import RealEnvironment  # 보상 계산용 (compute_reward 사용)

# ───────────────────────────────────────────────
# 설정
# ───────────────────────────────────────────────

@dataclass
class HyperParams:
    lr_q: float = 1e-3
    lr_r: float = 1e-3
    lr_s: float = 2e-3
    lr_alpha: float = 2e-3
    clip_step: float = 0.05
    save_interval_s: int = 6*3600  # 6시간마다 저장
    warmup_steps: int = 5
    max_q: float = 1e3
    max_r: float = 1e2
    max_s: float = 1e3
    alpha_bounds: Tuple[float,float] = (0.1, 10.0)

HP = HyperParams()

# ───────────────────────────────────────────────
# θ 로드 / 저장 (서버 θ 우선, 누락 자동 복구)
# ───────────────────────────────────────────────
def load_theta() -> dict:
    """서버 θ 기반 복구형 θ 로드"""

    # 폴더 경로
    server_dir = "server_trained"
    rpi_dir = "rpi_trained"

    os.makedirs(server_dir, exist_ok=True)
    os.makedirs(rpi_dir, exist_ok=True)

    server_path = os.path.join(server_dir, "trained_theta_server.pkl")
    local_path  = os.path.join(rpi_dir, "trained_theta.pkl")

    # 기본 θ (fallback)
    default_theta = {
        "Q": [2.0, 2.0, 0.0, 0.0],
        "R": [0.05, 0.05, 0.02],
        "S": [5.0, 5.0],
        "alpha_growth": 1.0,
    }

    # 1️⃣ 서버 pretrained θ 불러오기
    server_theta = default_theta.copy()
    if os.path.exists(server_path):
        try:
            with open(server_path, "rb") as f:
                tmp = pickle.load(f)
            if isinstance(tmp, dict):
                print(f"🌐 Loaded server pretrained θ from {server_path}")
                for k, v in default_theta.items():
                    if k not in tmp:
                        tmp[k] = v
                server_theta = tmp
        except Exception as e:
            print(f"⚠️ Failed to load server θ ({e}), using default fallback.")

    # 2️⃣ 로컬 θ 불러오기
    if not os.path.exists(local_path):
        print("⚠️ No RPi θ found → initializing from server θ")
        return server_theta

    try:
        with open(local_path, "rb") as f:
            theta = pickle.load(f)
        if not isinstance(theta, dict):
            print("⚠️ Invalid θ format, restored from server θ")
            return server_theta

        # 누락된 키 자동 보완
        for k, v in server_theta.items():
            if k not in theta:
                print(f"⚠️ Missing key '{k}' → restored from server θ")
                theta[k] = v

        print(f"✅ θ loaded successfully from {local_path}")
        return theta

    except Exception as e:
        print(f"⚠️ θ load failed ({e}) → restored from server θ")
        return server_theta


# ────────────────────────────────
# θ 저장 (Raspberry Pi 학습 결과 저장 전용)
# ────────────────────────────────
def save_theta(theta: dict):
    """RPi fine-tuned θ 저장 (rpi_trained 폴더에 저장)"""
    rpi_dir = "rpi_trained"
    os.makedirs(rpi_dir, exist_ok=True)

    path = os.path.join(rpi_dir, "trained_theta.pkl")
    tmp = path + ".tmp"

    try:
        with open(tmp, "wb") as f:
            pickle.dump(theta, f)
        os.replace(tmp, path)
        print(f"💾 θ saved → {path}")
    except Exception as e:
        print(f"❌ Failed to save θ: {e}")


# ───────────────────────────────────────────────
# MPC cost 근사 항 추정
# ───────────────────────────────────────────────
def estimate_terms(x: np.ndarray, crop: Dict, u: np.ndarray, u_prev: np.ndarray):
    temp, hum, co2, light = [float(v) for v in x]
    Tmin, Tmax = crop.get("target_temp", [18.0, 22.0])
    Hmin, Hmax = crop.get("target_humidity", [50.0, 70.0])
    Tref, Href = 0.5*(Tmin+Tmax), 0.5*(Hmin+Hmax)

    err_T = (temp - Tref)**2
    err_H = (hum  - Href)**2
    viol_T = max(0.0, Tmin-temp) + max(0.0, temp-Tmax)
    viol_H = max(0.0, Hmin-hum ) + max(0.0, hum -Hmax)
    viol = viol_T**2 + viol_H**2

    fan, heater, led = [float(np.clip(v,0,1)) for v in u]
    energy = fan**2 + heater**2
    du2 = float(np.sum((u - u_prev)**2))

    G_temp = math.exp(-0.5 * ((temp - 25.0) / 2.5) ** 2)
    G_hum  = math.exp(-0.5 * ((hum  - 60.0) / 8.0)  ** 2)
    G_light = math.tanh(light / 500.0)
    growth = G_temp * G_hum * G_light

    return dict(err_T=err_T, err_H=err_H, viol=viol,
                energy=energy, du2=du2, growth=growth)

# ───────────────────────────────────────────────
# RL 파라미터 업데이트 (Eq.22 근사)
# ───────────────────────────────────────────────
def update_theta(theta: Dict, terms: Dict[str,float]):
    Q = np.array(theta["Q"], dtype=float)
    R = np.array(theta["R"], dtype=float)
    S = np.array(theta["S"], dtype=float)
    alpha = float(theta["alpha_growth"])

    dq_T = (+HP.lr_q * terms["err_T"])
    dq_H = (+HP.lr_q * terms["err_H"])
    dS   = (+HP.lr_s * terms["viol"])
    dR   = (+HP.lr_r * (0.6*terms["du2"] + 0.4*terms["energy"]))
    dalpha = (+HP.lr_alpha * (terms["growth"] - 0.2)) - (HP.lr_alpha * 0.1 * (terms["viol"]>0))

    def step_clip(v, dv, vmax):
        delta = np.clip(dv, -HP.clip_step*max(1.0,abs(v)), HP.clip_step*max(1.0,abs(v)))
        return float(np.clip(v + delta, 0.0, vmax))

    if Q.size >= 2:
        Q[0] = step_clip(Q[0], dq_T, HP.max_q)
        Q[1] = step_clip(Q[1], dq_H, HP.max_q)
    if S.size >= 2:
        S[:] = [step_clip(v, dS, HP.max_s) for v in S]
    R[:] = [step_clip(v, dR, HP.max_r) for v in R]
    alpha = float(np.clip(alpha + dalpha, HP.alpha_bounds[0], HP.alpha_bounds[1]))

    theta["Q"], theta["R"], theta["S"], theta["alpha_growth"] = Q.tolist(), R.tolist(), S.tolist(), alpha

# ───────────────────────────────────────────────
# Main Loop
# ───────────────────────────────────────────────
def main():
    broker = "211.106.231.24"
    port = 1883
    farm = "farmA"
    esp = "esp1"

    env = RealEnvironment(broker_host=broker, broker_port=port, farm_id=farm, esp_id=esp)
    crop = env.crop
    theta = load_theta()
    u_prev = np.zeros(3)
    last_save = time.time()
    step = 0

    data = {"x": None, "d": None, "u": None}
    lock = threading.Lock()
    stop_flag = {"stop": False}

    def handle_sig(sig, frm):
        stop_flag["stop"] = True
        print("\n🛑 stopping...")
    import signal; signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except:
            return
        with lock:
            if msg.topic.endswith("/sensor"):
                data["x"] = np.array([
                    payload.get("temp_in",0.0),
                    payload.get("hum_in",0.0),
                    payload.get("co2_in",0.0),
                    payload.get("light_in",0.0)
                ], dtype=float)
            elif msg.topic.endswith("/disturbance"):
                data["d"] = np.array([
                    payload.get("solar_rad",0.0),
                    payload.get("co2_out",0.0),
                    payload.get("temp_out",0.0),
                    payload.get("hum_out",0.0)
                ], dtype=float)
            elif msg.topic.endswith("/mpc/u_opt"):
                data["u"] = np.array([
                    payload.get("fan",0.0),
                    payload.get("heater",0.0),
                    payload.get("led",0.0)
                ], dtype=float)

    # MQTT 재연결 콜백
    def on_disconnect(client, userdata, rc):
        print("⚠️ MQTT disconnected! Retrying...")
        while True:
            try:
                client.reconnect()
                print("🔁 MQTT reconnected successfully!")
                break
            except Exception as e:
                print(f"❌ Reconnect failed: {e}")
                time.sleep(5)


    # MQTT 연결
    client = mqtt.Client()
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    while True:
        try:
            client.connect(broker, port, keepalive=30)
            print(f"✅ MQTT connected → {broker}:{port}")
            break
        except Exception as e:
            print(f"❌ MQTT connection failed: {e}")
            print("🔁 5초 후 다시 시도합니다...")
            time.sleep(5)
            
    client.subscribe([
        (f"{farm}/{esp}/sensor", 0),
        (f"{farm}/{esp}/disturbance", 0),
        (f"{farm}/{esp}/mpc/u_opt", 0),
    ])
    client.loop_start()
    print(f"✅ MQTT connected, subscribed to topics for {farm}/{esp}")

    print("🚀 RL Learner (MQTT-based) started — learning only\n")

    # ───────────────────────────────────────────────
    # RL 학습 루프
    # ───────────────────────────────────────────────

    while not stop_flag["stop"]:
        time.sleep(env.sample_time)
        with lock:
            x, d, u = data["x"], data["d"], data["u"]
        if x is None or u is None:
            continue

        # 보상 계산 및 파라미터 업데이트
        r = env.compute_reward(x, u, u_prev)
        if step >= HP.warmup_steps:
            terms = estimate_terms(x, crop, u, u_prev)
            update_theta(theta, terms)
        u_prev = u.copy()
        step += 1

        if (time.time() - last_save) >= HP.save_interval_s:
            save_theta(theta, DEFAULT_THETA_PATH)
            last_save = time.time()

    client.loop_stop()
    save_theta(theta, DEFAULT_THETA_PATH)
    print("✅ RL learning-only (MQTT) finished.")

if __name__ == "__main__":
    main()
