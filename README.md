# 🌱 Reinforcement Learning-Based Model Predictive Control for Greenhouse Climate Control

> **Real-World Implementation of RL-MPC (CasADi-based, Paper Equations (18)–(21))**

This repository provides a **real-environment version** of the RL-MPC algorithm proposed in  
[Mallick et al., *Smart Agricultural Technology*, 2025](https://doi.org/10.1016/j.atech.2024.100751).  
It integrates a **CasADi-based Model Predictive Controller (MPC)** with **Reinforcement Learning (Q-learning-style parameter adaptation)**, communicating through **MQTT** to control greenhouse actuators in real time.

---

## 📂 Project Structure

```
├── main_real_detail.py          # Main RL-MPC execution loop (real environment)
├── learning_real_detail.py      # CasADi MPC + RL parameter update (θ = {Q,R,S,α})
├── real_env.py                  # MQTT I/O adapter for real greenhouse sensors & actuators
├── lettuce_profile.json         # Crop-specific environmental target profile (example: lettuce)
├── trained_theta.pkl            # Stored RL parameters (updated every 6 hours)
└── logs/                        # Auto-generated log files
```

---

## ⚙️ System Overview

### 🔸 1. Real Environment (MQTT I/O)
The **`RealEnvironment`** class connects to actual devices via MQTT:

| Topic | Direction | Example |
|-------|------------|---------|
| `{farm_id}/{esp_id}/sensor` | ESP → Controller | Temperature, humidity, CO₂, light |
| `{farm_id}/{esp_id}/disturbance` | ESP → Controller | Solar radiation, outdoor temp/hum |
| `{farm_id}/{esp_id}/actuator/control` | Controller → ESP | Fan, heater, LED |

**Default settings**
```python
broker_host = "172.27.148.207"
broker_port = 1883
farm_id     = "farmA"
esp_id      = "esp1"
sample_time = 5.0   # seconds
```

---

### 🔸 2. CasADi MPC Core (`LearningMpcCasADi`)

Implements the parametric MPC from the paper’s formulation (Eq. 18 – 21).  
State `x=[temp, hum, co₂, light]`, control `u=[fan, heater, led]`, disturbance `d=[rad, co₂_out, outT, outH]`.

#### Objective Function  
\[
J = \sum_{k=0}^{N-1} \Big[(x_k-r)^T Q (x_k-r) + (\Delta u_k)^T R (\Delta u_k)
+ u_k^T S u_k + J_{slack} - \alpha_{growth} G(x_k)\Big]
\]

where  
- \(J_{slack}\): constraint violation penalty  
- \(G(x_k)\): crop growth contribution  
- \(Q,R,S,\alpha_{growth}\) = RL-learnable parameters (stored in `trained_theta.pkl`)

The solver uses **CasADi SQPMethod** for real-time optimization on Raspberry Pi.

---

### 🔸 3. RL Parameter Learning (Q-Learning Style)

The function `update_theta()` performs a simplified **Second-Order LSTD Q-Learning** update
based on replay buffer tuples \((s, u, r, s')\):

\[
\theta \leftarrow \theta - \eta\nabla_\theta \tfrac{1}{2}\delta^2,
\quad
\delta = r + \gamma V_\theta(s') - Q_\theta(s,u)
\]

Parameters \(Q, R, S, \alpha_{growth}\) are adjusted every **6 hours** (`UPDATE_PERIOD = 6 h`).

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install numpy casadi paho-mqtt torch
```

### 2️⃣ Run the RL-MPC controller
```bash
python main_real_detail.py
```

### 3️⃣ Console output example
```
✅ MQTT connected → 172.27.148.207:1883
🎯 Set references → T_ref=25.0°C, H_ref=60.0%
✅ RL-MPC loop running...

⏱ [STEP 023] t=1234.56
🏆 Reward=-1.247 | Track=0.412 Δu=0.033 Slack=0.000 Energy=0.052 Growth=0.893
⚙️ u_opt → FAN=0.41 | HEATER=0.32 | LED=0.50
```

Log files are automatically stored in `logs/rl_mpc_YYYYMMDD_HHMMSS.log`.

---

## 🌾 Crop Profile Example (`lettuce_profile.json`)

```json
{
  "crop_name": "lettuce",
  "description": "상추 생육 최적 환경 설정",
  "targets": { "temperature": 25.0, "humidity": 0.65 },
  "priority": { "temperature": 1.0, "humidity": 0.7 },
  "control_weights": { "fan": 0.05, "heater": 0.02, "led": 0.01 }
}
```

Used by `RealEnvironment` to set MPC reference values  
→ `T_ref`, `H_ref`, and reward shaping terms.

---

## 🧠 Core Algorithm Flow

```text
┌────────────────────────────────────────────────────────────┐
│                     Real-World RL-MPC                      │
├────────────────────────────────────────────────────────────┤
│ 1. Read sensors via MQTT (x, d)                            │
│ 2. MPC (CasADi) computes optimal control u_opt             │
│ 3. Send actuator commands (fan, heater, led)               │
│ 4. Compute reward r = -(J_track+J_delta+J_slack+J_energy)  │
│                      + α_growth * G(x)                     │
│ 5. Store (s, u, r, s') in replay buffer                    │
│ 6. Every 6 h → update θ = {Q,R,S,α_growth} via Q-learning  │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 Reward Composition

| Term | Symbol | Meaning |
|------|---------|----------|
| Tracking error | \(J_{track}\) | Deviation from crop temperature & humidity targets |
| Control rate | \(J_{Δu}\) | Penalty for large actuator changes |
| Slack | \(J_{slack}\) | Constraint violation (temperature / humidity bounds) |
| Energy | \(J_{energy}\) | Fan + Heater + LED consumption cost |
| Growth | \(G(x)\) | Biological growth contribution (Gaussian × tanh) |

\[
r = - (J_{track}+J_{Δu}+J_{slack}+J_{energy}) + c_{growth}·G(x)
\]

---

## 💾 RL Parameter File (`trained_theta.pkl`)

Saved every update cycle:
```python
{
  "Q": [ ... ],
  "R": [ ... ],
  "S": [ ... ],
  "alpha_growth": 1.0
}
```
Reloaded automatically at startup.

---

## 🧩 Hardware Integration

| Component | Role |
|------------|------|
| **ESP32 Node(s)** | Publishes sensor data and receives control signals via MQTT |
| **Raspberry Pi (Controller)** | Runs CasADi + PyTorch RL-MPC loop |
| **Mosquitto Broker** | Local MQTT hub (port 1883) |
| **Cloud Dashboard (optional)** | For remote logging, OTA, or visualization |

---

## 🧾 Reference

> **S. Mallick, F. Airaldi, A. Dabiri, C. Sun, B. De Schutter**,  
> *Reinforcement Learning-Based Model Predictive Control for Greenhouse Climate Control*,  
> *Smart Agricultural Technology*, Vol 10, 2025, 100751.  
> [https://doi.org/10.1016/j.atech.2024.100751](https://doi.org/10.1016/j.atech.2024.100751)

---

## 📘 Citation

```bibtex
@article{mallick2025mpcrl,
  title={Reinforcement Learning-Based Model Predictive Control for Greenhouse Climate Control},
  author={Mallick, Samuel and Airaldi, Filippo and Dabiri, Azita and Sun, Congcong and De Schutter, Bart},
  journal={Smart Agricultural Technology},
  volume={10},
  year={2025},
  pages={100751},
  doi={10.1016/j.atech.2024.100751}
}
```

---

## 🧑‍💻 Author Notes

- This repository adapts the original **simulation-based** RL-MPC into a **real-environment control system**.
- Designed for **on-device AI deployment** on Raspberry Pi + ESP32 network.
- CasADi, PyTorch, and MQTT form the minimal real-time control stack.

---

**© 2025 Greenhouse RL-MPC Research — Open Use for Academic and Smart Farm Integration**
