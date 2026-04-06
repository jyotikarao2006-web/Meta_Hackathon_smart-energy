---
title: Smart Building Energy Management RL Environment
emoji: ⚡
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

# ⚡ Smart Energy Management — RL Environment v2.0

A competition-grade Reinforcement Learning environment for smart building energy control.

An AI agent controls **HVAC** (3 zones), a **lithium battery**, and **solar export** to minimise electricity costs while keeping occupants comfortable.

---

## 🆕 v2.0 Differentiators

| Feature | v1 | v2 |
|---------|----|----|
| Electricity pricing | Binary peak/off-peak | **Dynamic real-time curve** (24-hour spot market) |
| Battery | Simple | **Degradation model** (cycle counting, capacity fade) |
| HVAC | Single zone | **3 zones**: office · server room · lobby |
| Weather | Fixed sine wave | **Stochastic daily character** (clear / cloudy / overcast) |
| Reward | 4 components | **8 components** with partial-progress signals |
| Observation | 7 features | **11 features** (zone temps, humidity, wind, battery health) |

---

## 🌐 API (OpenEnv v1.1)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode → `EnergyState` |
| `POST` | `/step` | Take action → `StepResult` |
| `GET`  | `/state` | Current state (no side effects) |
| `GET`  | `/health` | `{"status": "ok", "version": "2.0.0"}` |
| `GET`  | `/tasks` | List all tasks |
| `POST` | `/grade` | Grade an episode → `{"score": 0.xx}` |

### Quick start

```bash
# Reset
curl -X POST http://localhost:7860/reset

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"hvac_setpoint": 0.4, "battery_action": 0.7}}'

# State
curl http://localhost:7860/state
```

---

## 📐 Observation Space (11 features)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hour` | int | 0–23 | Hour of day |
| `indoor_temp` | float | 10–45 °C | Office zone temperature |
| `outdoor_temp` | float | 10–45 °C | Ambient temperature |
| `battery_level` | float | 0–10 kWh | Current charge |
| `battery_capacity` | float | 8–10 kWh | Effective capacity (post-degradation) |
| `battery_health` | float | 0.80–1.0 | Health fraction |
| `solar_power` | float | 0–4 kW | Real-time solar output |
| `grid_price` | float | $0.05–0.40/kWh | Dynamic real-time price |
| `is_peak_hour` | bool | — | True when price ≥ $0.25/kWh |
| `total_cost` | float | USD | Cumulative cost this episode |
| `comfort_violations` | int | ≥0 | Zone-hours outside 20–26 °C |
| `zone_temps` | dict | — | `{office, server_room, lobby}` temps |
| `humidity` | float | 20–95% | Outdoor relative humidity |
| `wind_speed` | float | 0–8 m/s | Wind speed |

### Normalised vector (7 features for RL algorithms)

```
[hour/24, indoor_temp/40, outdoor_temp/40, battery_pct,
 solar_power/4, grid_price/0.35, is_peak_hour]
```

---

## 🎮 Action Space (continuous)

| Field | Range | Description |
|-------|-------|-------------|
| `hvac_setpoint` | [0.0, 1.0] | Aggregate HVAC power (0=off, 1=full 3.5 kW) |
| `battery_action` | [-1.0, 1.0] | -1=max discharge, 0=hold, +1=max charge |

---

## 🏆 Tasks

### 🟢 Task 1 — Peak Shaving (Easy)
Minimise 24-hour grid cost using battery + solar to avoid peak-hour purchases.

| Cost | Score |
|------|-------|
| ≥ $9.00 | 0.00 |
| $6.00 | 0.30 |
| $3.50 | 0.70 |
| ≤ $2.20 | 1.00 |

Comfort violations: −0.02/hour, capped at −0.15.

### 🟡 Task 2 — Comfort First (Medium)
Maintain all zones at 20–26 °C while minimising cost. Violations cap the score.

| Violations | Max Score |
|-----------|-----------|
| > 0 | `max(0, 0.50 − violations × 0.05)` |
| 0, cost ≥ $8 | 0.30 |
| 0, cost = $4 | 0.65 |
| 0, cost ≤ $2.20 | 1.00 |

### 🔴 Task 3 — Solar Arbitrage (Hard)
Three equal sub-scores: grid cost + solar sold back + zero violations.

| Sub-score | Weight | Target |
|-----------|--------|--------|
| Cost | 0.33 | $9→0.0, $2.20→0.33 |
| Solar sold back | 0.34 | 0→0.0, 9 kWh→0.34 |
| Comfort | 0.33 | 0 violations→0.33 |

---

## 🎯 Reward Design

Dense, multi-factor reward (8 components):

```python
reward = (
    - hourly_cost * 10.0                    # cost penalty (continuous)
    + (ok_zones / total_zones) * 2.0        # comfort partial credit
    - sum(|T - bound| * 0.5 per violation)  # temperature distance
    + solar_used_kw * 0.4                   # solar utilisation
    + 0.3 if charging_with_solar            # smart charging bonus
    + 0.4 if discharging_at_peak            # peak discharge bonus
    + 0.2 if 25% <= battery_pct <= 75%      # healthy SoC
    + battery_health * 0.05                 # health preservation
)
```

---

## 🏃 Running Locally

```bash
# Install
pip install -r requirements.txt

# Validate (no server needed)
python validate.py

# Web dashboard + API
python app.py
# → http://localhost:7860

# LLM inference (requires API key)
export HF_TOKEN=your_key
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

## 🐳 Docker

```bash
docker build -t smart-energy .
docker run -p 7860:7860 smart-energy
# Dashboard: http://localhost:7860

# Run inference
docker run -e HF_TOKEN=xxx -e MODEL_NAME=gpt-4o-mini \
  smart-energy python inference.py
```

---

## 🧪 Baseline Scores (seed=42, 2 episodes each)

| Agent | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) |
|-------|--------------|----------------|--------------|
| Random | 0.08 | 0.31 | 0.47 |
| Rule-Based | 0.57 | 0.62 | 0.60 |
| Solar Arbitrage | 0.62 | 0.65 | 0.84 |

---

## ⚡ Performance

- Episode: ~1 ms (CPU, no GPU required)
- Full inference run (3 tasks × 2 episodes): **< 5 min** on 2-CPU machine
- RAM footprint: **< 50 MB**

---

## 📁 Project Structure

```
├── env/
│   └── energy_env.py       # Core environment (weather, battery, HVAC)
├── models/
│   └── schemas.py          # Pydantic typed models
├── tasks/
│   └── task_definitions.py # 3 graded tasks
├── app.py                  # Flask API + dashboard
├── inference.py            # LLM agent runner
├── validate.py             # Smoke-test suite
├── openenv.yaml            # OpenEnv spec v1.1
├── pyproject.toml
├── requirements.txt
└── Dockerfile
```
