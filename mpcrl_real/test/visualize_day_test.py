"""
visualize_day_test.py
---------------------
MPC의 하루 제어 테스트 결과를 자동으로 로그 + 그래프로 시각화.
ESP 시뮬레이터 MQTT 데이터를 받아 MPC를 실행하며,
센서값, MPC 예측값, 목표값을 동시에 표시하고 저장.
"""

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from real_env import RealEnvironment
from default_real import DefaultReal
from learning_real import LearningMpcReal

# 저장 폴더
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"mpc_daytest_{int(time.time())}.csv")


def write_csv_header():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "temp_sensor", "temp_pred", "temp_ref",
            "humidity_sensor", "radiation", "fan", "heater", "led"
        ])


def append_log(timestamp, x, x_pred, x_ref, d, u):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            x[2], x_pred, x_ref,
            x[1], d[0], u[0], u[1], u[2]
        ])


def visualize_day_test(duration_hours=24, speed_factor=60):
    """
    duration_hours : 실험 시간 (24h 시뮬레이션)
    speed_factor   : 60이면 1초가 1분에 해당 (빠른 배속)
    """
    print("🚀 Starting 24-hour MPC day test visualization...")

    env = RealEnvironment(broker_ip="localhost", farm_id="farmA", esp_id="esp1", sample_time=1.0)
    test = DefaultReal()
    mpc = LearningMpcReal(test=test)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.title("MPC Day Test - Temperature Control")
    plt.xlabel("Time (virtual hours)")
    plt.ylabel("Normalized Temperature (0~1)")
    line_sensor, = ax.plot([], [], 'bo-', label="Sensor")
    line_pred,   = ax.plot([], [], 'r--', label="Predicted by MPC")
    line_ref,    = ax.plot([], [], 'g-', label="Reference (Target)")
    ax.legend()

    write_csv_header()

    x_ref = 0.5  # 목표 온도 (25°C)
    t = 0.0
    times, temp_sensors, temp_preds, temp_refs = [], [], [], []

    try:
        while t < duration_hours:
            x = env.read_sensors()
            d = env.read_disturbance()

            # MPC 계산
            u_opt, status = mpc.compute_control(x, d)

            # 예측된 다음 상태
            x_pred_next = mpc.model.rk4_step(x, u_opt, d, np.ones(4), test.ts)
            temp_pred = x_pred_next[2]

            # 로깅
            append_log(time.time(), x, temp_pred, x_ref, d, u_opt)

            # 그래프 갱신
            times.append(t)
            temp_sensors.append(x[2])
            temp_preds.append(temp_pred)
            temp_refs.append(x_ref)

            ax.clear()
            ax.plot(times, temp_sensors, 'bo-', label="Sensor")
            ax.plot(times, temp_preds, 'r--', label="Predicted by MPC")
            ax.plot(times, temp_refs, 'g-', label="Target (x_ref=0.5)")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, duration_hours)
            ax.set_xlabel("Virtual Time (hours)")
            ax.set_ylabel("Normalized Temperature (0~1)")
            ax.legend()
            plt.pause(0.01)

            env.apply_control(u_opt)
            env.wait_next_cycle()

            t += 1.0 / speed_factor  # 가속된 시간 (1초 = 1분)
    except KeyboardInterrupt:
        print("🛑 Test stopped manually.")
    finally:
        print(f"✅ Log saved at: {LOG_FILE}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    visualize_day_test(duration_hours=24, speed_factor=60)
