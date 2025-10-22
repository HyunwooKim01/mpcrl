"""
mpc_test_once.py
----------------
MPC 미래 예측 결과를 단회 테스트하는 스크립트.
- 센서 입력(x_current), 외란(d_current)을 임의로 지정
- MPC가 예측한 미래 상태(x_pred), 제어 입력(u_opt) 전체 출력
"""

import numpy as np
from sims.configs.default_real import DefaultReal
from learning_real import LearningMpcReal
from greenhouse.model_real import Model

print("🚀 MPC 단회 테스트 시작")

# ------------------------------
# 1️⃣ 테스트용 입력 데이터 설정
# ------------------------------
# 실제 센서로부터 들어올 수 있는 정규화된 값 (0~1 기준)
x_current = np.array([0.003, 0.45, 0.20, 0.0075])

# 외란 입력 (정규화 또는 비율 기반)
d_current = np.array([
    0.5,   # radiation (중간 정도 햇빛)
    0.3,   # 외부 CO2 or 기타 외란
    0.6,   # 외기온 영향
    0.7    # 외기습도 영향
])

# ------------------------------
# 2️⃣ MPC 초기화
# ------------------------------
config = DefaultReal("lettuce")
mpc = LearningMpcReal(test=config)
print("✅ MPC controller initialized.\n")

# ------------------------------
# 3️⃣ MPC 계산 수행
# ------------------------------
u_opt, status = mpc.compute_control(x_current, d_current)
print(f"[MPC STATUS] {status}")
print(f"[u_opt] {u_opt}\n")

# ------------------------------
# 4️⃣ 내부 예측 trajectory 추출
# ------------------------------
# N-step 예측을 수행한 뒤, 내부적으로 계산된 상태 궤적 추출
params = {**config.learnable_pars_init, **config.fixed_pars}
params["d"] = np.tile(d_current.reshape(-1, 1), (1, mpc.prediction_horizon))
params["x_0"] = x_current.reshape(-1, 1)

sol = mpc.solve(pars=params, vals0={"x": x_current.reshape(-1, 1)})

x_pred = np.array(sol.vals["x"])  # (nx, N+1)
u_pred = np.array(sol.vals["u"])  # (nu, N)

# ------------------------------
# 5️⃣ 결과 출력
# ------------------------------
print("📈 [MPC 미래 예측 Trajectory]")
print("Step | Temperature | Humidity | Biomass | LeafWater | Fan | Heater | LED")
print("--------------------------------------------------------------")

for k in range(mpc.prediction_horizon):
    xk = x_pred[:, k]
    uk = u_pred[:, k] if k < u_pred.shape[1] else np.zeros(3)
    print(f"{k:>3d}  | {xk[2]:.4f} | {xk[1]:.4f} | {xk[0]:.4f} | {xk[3]:.4f} | "
          f"{uk[0]:.3f} | {uk[1]:.3f} | {uk[2]:.3f}")

print("\n✅ MPC 예측 테스트 완료.")
