# 데이터 확인용 코드
import pickle

with open("test_80_train.pkl", "rb") as f:
    data = pickle.load(f)

# 각 변수의 데이터를 출력해서 실제 내용 확인
print("X (States):", data["X"])
print("U (Control Inputs):", data["U"])
print("R (Rewards):", data["R"])

# 각 항목을 확인해보며 그래프에 반영되는 데이터가 제대로 있는지 확인
