import numpy as np
import pickle
import matplotlib.pyplot as plt

# 데이터 로딩
with open("test_80_train.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("test_80_eval.pkl", "rb") as f:
    eval_data = pickle.load(f)

# 훈련 데이터와 평가 데이터에서 보상값 추출
reward_train = np.array(train_data["R"])  # shape: (episodes, steps)
reward_eval = np.array(eval_data["R"])   # shape: (episodes, steps)

# 보상 값의 평균 구하기
avg_reward_train = reward_train.mean(axis=0)  # 각 스텝에 대해 평균
avg_reward_eval = reward_eval.mean(axis=0)

# 시간 축 설정
time_train = np.arange(len(avg_reward_train))
time_eval = np.arange(len(avg_reward_eval))

# 그래프 생성
fig, ax = plt.subplots(figsize=(10, 6))

# 평균 보상 그래프
ax.plot(time_train, avg_reward_train, label="Avg Reward - Train", color='blue')
ax.plot(time_eval, avg_reward_eval, label="Avg Reward - Eval", color='green')

ax.set_title("Average Reward Over Time")
ax.set_xlabel("Time (each 1 growth episode)")
ax.set_ylabel("Average Reward")
ax.legend()

plt.tight_layout()
plt.show()
