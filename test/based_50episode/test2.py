import numpy as np
import pickle
import matplotlib.pyplot as plt

# 데이터 로딩
with open("test_80_train.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("test_80_eval.pkl", "rb") as f:
    eval_data = pickle.load(f)

# 보상값 추출
reward_train = np.array(train_data["R"])
reward_eval = np.array(eval_data["R"])

# 평균 보상 계산
avg_reward_train = reward_train.mean(axis=0)
avg_reward_eval = reward_eval.mean(axis=0)

# 이동평균 적용 함수
def smooth(data, window_size=20):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

# 곡선 smoothing (윈도우 크기 조절 가능)
smooth_train = smooth(avg_reward_train, window_size=3)
smooth_eval = smooth(avg_reward_eval, window_size=3)

# 시간 축
time_train = np.arange(len(avg_reward_train))
time_eval = np.arange(len(avg_reward_eval))

# 그래프
plt.figure(figsize=(10, 6))
plt.plot(time_train, smooth_train, label="Smoothed Reward - Train", color='blue')
plt.plot(time_eval, smooth_eval, label="Smoothed Reward - Eval", color='green', alpha=0.8)

plt.title("Average Reward Over Time (Smoothed)")
plt.xlabel("Time (each 1 growth episode)")
plt.ylabel("Average Reward")
plt.legend()
plt.tight_layout()
plt.show()
