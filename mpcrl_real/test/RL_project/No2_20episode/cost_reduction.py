# save as: cost_reduction.py
import pickle, numpy as np, matplotlib.pyplot as plt

TRAIN = "test_80_train.pkl"
EVAL  = "test_80_eval.pkl"
SMOOTH = 50  # 이동평균 window (에피소드 곡선용)

def load_R(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["R"])  # shape: (episodes, steps)

def smooth(x, w=50):
    if w <= 1: return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

# 1) 데이터 로드
R_tr = load_R(TRAIN)
R_ev = load_R(EVAL)

# 2) (삭제됨) 특정 step 구간 제외 없이 전체 구간 사용

# 3) Cost 계산 (= -Reward)
C_tr_ep = -R_tr.sum(axis=1)  # 각 episode의 total cost
C_ev_ep = -R_ev.sum(axis=1)

# 4) Cost 절감율 계산
def reduction_rate(cost_array):
    start, end = cost_array[0], cost_array[-1]
    reduction = (start - end) / abs(start) * 100  # 필요시 양수표기: (abs(start)-abs(end))/abs(start)*100
    return reduction, start, end

train_reduction, train_start, train_end = reduction_rate(C_tr_ep)
eval_reduction, eval_start, eval_end   = reduction_rate(C_ev_ep)

print("===== 📊 Cost Reduction Summary =====")
print(f"[Train] 시작 cost: {train_start:,.2f}, 종료 cost: {train_end:,.2f}")
print(f"[Train] 절감율: {train_reduction:.2f}%")
print(f"[Eval ] 시작 cost: {eval_start:,.2f}, 종료 cost: {eval_end:,.2f}")
print(f"[Eval ] 절감율: {eval_reduction:.2f}%")
print("====================================")

# 5) 시각화
plt.figure(figsize=(11,5))
plt.plot(smooth(C_tr_ep, 5), label="Cost/episode (Train)")
plt.plot(smooth(C_ev_ep, 5), label="Cost/episode (Eval)", alpha=0.9)
plt.title("Total Cost per Episode (Full Range)")
plt.xlabel("Episode"); plt.ylabel("Total Cost")
plt.legend(); plt.tight_layout(); plt.show()
