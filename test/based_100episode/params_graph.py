import pickle, numpy as np, matplotlib.pyplot as plt

with open("test_80_train.pkl", "rb") as f:
    data = pickle.load(f)

X = np.asarray(data["X"])   # (episodes, steps_x, x_dim)
U = np.asarray(data["U"])   # (episodes, steps_u, u_dim)
R = np.asarray(data["R"])   # (episodes, steps_r)

E, steps_x, _ = X.shape
_, steps_u, _ = U.shape
_, steps_r    = R.shape

# 시간축을 '각 데이터 길이'에 맞춰 따로 생성
t_x = np.arange(steps_x)       # 상태(온도/습도)용
t_u = np.arange(steps_u)       # 제어입력용
t_r = np.arange(steps_r)       # 보상용

# 에피소드 평균 시계열
mean_X = X.mean(axis=0)        # (steps_x, x_dim)
mean_U = U.mean(axis=0)        # (steps_u, u_dim)
mean_R = R.mean(axis=0)        # (steps_r,)

# 컬럼 매핑 (당신 로그 기준)
T_in     = mean_X[:, 2]        # 실내 온도
Humidity = mean_X[:, 3]        # 습도(절대습도 계열)
Heating  = mean_U[:, 0]
Cooling  = mean_U[:, 1]
CO2      = mean_U[:, 2]
T_ref    = np.full_like(T_in, 15.0)  # 목표 온도(알고 있다면 교체)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1) 온도
axs[0, 0].plot(t_x, T_in,  label="Indoor Temp (T_in)")
axs[0, 0].plot(t_x, T_ref, "--", label="Reference Temp (T_ref)")
axs[0, 0].set_title("Indoor Temperature vs Reference")
axs[0, 0].set_xlabel("Time steps")
axs[0, 0].set_ylabel("Temperature (°C)")
axs[0, 0].legend()

# 2) 습도
axs[0, 1].plot(t_x, Humidity, label="Humidity")
axs[0, 1].set_title("Humidity Over Time")
axs[0, 1].set_xlabel("Time steps")
axs[0, 1].set_ylabel("Humidity")
axs[0, 1].legend()

# 3) 제어입력
axs[1, 0].plot(t_u, Heating, label="Heating")
axs[1, 0].plot(t_u, Cooling, label="Cooling")
axs[1, 0].plot(t_u, CO2,    label="CO₂")
axs[1, 0].set_title("Control Inputs")
axs[1, 0].set_xlabel("Time steps")
axs[1, 0].set_ylabel("Input")
axs[1, 0].legend()

# 4) 보상
axs[1, 1].plot(t_r, mean_R, label="Reward")
axs[1, 1].set_title("Reward Over Time")
axs[1, 1].set_xlabel("Time steps")
axs[1, 1].set_ylabel("Reward")
axs[1, 1].legend()

plt.tight_layout(pad=3)
plt.savefig("visualization_based.png", dpi=300)
print("Saved: visualization_based.png")
plt.show()