"""
inference.py — Smart Energy Management RL  v2.0
================================================
LLM-powered agent that plays all three OpenEnv tasks using an OpenAI-compatible API.

Required environment variables:
  API_BASE_URL   LLM endpoint   (default: https://api.openai.com/v1)
  MODEL_NAME     Model name     (default: gpt-4o-mini)
  HF_TOKEN       API key

Output format (strict):
  [START] task_id=<id> task_name=<name> difficulty=<level> episode=<n>
  [STEP]  step=<n> action=hvac=<v>/bat=<v> indoor_temp=<T>C battery=<b>kWh ...
  [END]   task_id=<id> episode=<n> score=<0-1> cost=<$> violations=<n> ...

Run:
    python inference.py
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from env.energy_env import SmartEnergyEnv
from models.schemas import EnergyAction
from tasks.task_definitions import TASKS, TaskStats

# ── Configuration ─────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     " ")
SEED         = 42
NUM_EPISODES = 2    # 2 episodes per task ≈ 6-8 min total on CPU

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set — using empty API key.", flush=True)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert energy management controller for a smart building with:
  • 3 HVAC zones (office, server room, lobby)
  • A 10 kWh lithium battery with degradation tracking
  • 4 kW solar array with weather-dependent output
  • Dynamic real-time electricity pricing (not just binary peak/off-peak)

You control two actuators each hour:
  • hvac_setpoint  [0.0–1.0]   HVAC power fraction (0=off, 1=full 3.5 kW aggregate)
  • battery_action [-1.0–1.0]  battery (-1=max discharge, 0=hold, +1=max charge)

Respond ONLY with a valid JSON object, nothing else:
{"hvac_setpoint": 0.4, "battery_action": 0.6}"""


def build_user_prompt(state, task_prompt: str, step: int) -> str:
    zone_info = ""
    if hasattr(state, "zone_temps") and state.zone_temps:
        zone_info = "- Zone temps:         " + ", ".join(
            f"{z}={t:.1f}°C" for z, t in state.zone_temps.items()
        ) + "\n"
    return f"""Task instructions:
{task_prompt}

Current building state (hour {state.hour}/24, step {step}):
- Hour of day:         {state.hour}  ({'PEAK' if state.is_peak_hour else 'off-peak'} — price ${state.grid_price:.4f}/kWh)
- Indoor temp (office):{state.indoor_temp:.1f} °C  (comfort: 20–26 °C)
- Outdoor temp:        {state.outdoor_temp:.1f} °C
{zone_info}- Battery:             {state.battery_level:.2f}/{state.battery_capacity:.1f} kWh  ({100*state.battery_level/max(state.battery_capacity,0.01):.0f}%)  health={getattr(state,'battery_health',1.0):.3f}
- Solar output:        {state.solar_power:.2f} kW
- Grid price:          ${state.grid_price:.4f}/kWh  (dynamic real-time pricing)
- Cost so far:         ${state.total_cost:.4f}
- Comfort violations:  {state.comfort_violations} zone-hours
- Humidity:            {getattr(state,'humidity',50):.0f}%  wind={getattr(state,'wind_speed',1):.1f} m/s

Decide your action. Output ONLY JSON:
{{"hvac_setpoint": <0.0–1.0>, "battery_action": <-1.0 to 1.0>}}"""


def llm_agent(state, task_prompt: str, step: int) -> EnergyAction:
    """Query LLM; fall back to rule-based heuristic on error."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(state, task_prompt, step)},
            ],
            max_tokens=64,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        return EnergyAction(
            hvac_setpoint=float(data.get("hvac_setpoint", 0.5)),
            battery_action=float(data.get("battery_action", 0.0)),
        )
    except Exception as exc:
        print(f"[WARN] LLM call failed ({exc}), using rule-based fallback", flush=True)
        return _rule_based_fallback(state)


def _rule_based_fallback(state) -> EnergyAction:
    bat_pct = state.battery_level / max(state.battery_capacity, 0.01)
    if not state.is_peak_hour and state.solar_power > 1.0:
        bat = 0.9
    elif state.is_peak_hour and bat_pct > 0.3:
        bat = -0.8
    elif bat_pct < 0.2:
        bat = 0.4
    else:
        bat = 0.0
    err = abs(state.indoor_temp - 22.0)
    hvac = 0.2 if err < 1.0 else (0.5 if err < 3.0 else 0.8)
    if state.is_peak_hour:
        hvac *= 0.6
    return EnergyAction(hvac_setpoint=hvac, battery_action=bat)


def run_episode(env: SmartEnergyEnv, task_id: str, task, episode: int) -> float:
    state        = env.reset()
    total_reward = 0.0
    step         = 0

    print(
        f"[START] task_id={task_id} task_name={task.name!r} "
        f"difficulty={task.difficulty} episode={episode}",
        flush=True,
    )

    while True:
        action = llm_agent(state, task.prompt, step)
        result = env.step(action)
        total_reward += result.reward
        step  += 1
        state  = result.state

        interim_stats = TaskStats(
            total_cost=state.total_cost,
            comfort_violations=state.comfort_violations,
            solar_generated=env.total_solar_generated,
            solar_sold_back=env.total_solar_sold_back,
            total_reward=total_reward,
            steps=step,
        )
        running_score = task.grade(interim_stats)

        print(
            f"[STEP] step={step} "
            f"action=hvac={action.hvac_setpoint:.3f}/bat={action.battery_action:.3f} "
            f"indoor_temp={state.indoor_temp:.1f}C "
            f"battery={state.battery_level:.2f}kWh "
            f"health={getattr(state,'battery_health',1.0):.3f} "
            f"solar={result.info.get('solar_power_kw',0.0):.2f}kW "
            f"price=${result.info.get('grid_price',0):.4f}/kWh "
            f"cost_step=${result.info.get('hourly_cost',0.0):.4f} "
            f"cost_total=${state.total_cost:.4f} "
            f"violations={state.comfort_violations} "
            f"reward={result.reward:.4f} "
            f"score={running_score:.4f} "
            f"done={result.done}",
            flush=True,
        )

        if result.done:
            break

    final_stats = TaskStats(
        total_cost=state.total_cost,
        comfort_violations=state.comfort_violations,
        solar_generated=env.total_solar_generated,
        solar_sold_back=env.total_solar_sold_back,
        total_reward=total_reward,
        steps=step,
    )
    final_score = task.grade(final_stats)

    print(
        f"[END] task_id={task_id} episode={episode} "
        f"score={final_score:.4f} "
        f"cost=${state.total_cost:.4f} "
        f"violations={state.comfort_violations} "
        f"solar_generated={env.total_solar_generated:.2f}kWh "
        f"solar_sold_back={env.total_solar_sold_back:.3f}kWh "
        f"battery_health={getattr(state,'battery_health',1.0):.4f} "
        f"total_reward={total_reward:.4f} "
        f"steps={step}",
        flush=True,
    )

    return final_score


def main():
    print("=" * 72, flush=True)
    print("  Smart Energy RL v2.0 — LLM Inference", flush=True)
    print(f"  model={MODEL_NAME}  api_base={API_BASE_URL}", flush=True)
    print(f"  seed={SEED}  episodes_per_task={NUM_EPISODES}", flush=True)
    print(f"  features: dynamic pricing · battery degradation · multi-room HVAC · weather", flush=True)
    print("=" * 72, flush=True)

    random.seed(SEED)
    env = SmartEnergyEnv()
    all_scores: dict = {}

    for task_id, task in TASKS.items():
        scores = []
        for ep in range(1, NUM_EPISODES + 1):
            random.seed(SEED + ep)
            score = run_episode(env, task_id, task, episode=ep)
            scores.append(score)
        all_scores[task_id] = scores

    print("\n" + "=" * 72, flush=True)
    print("  FINAL SCORES (averaged over episodes)", flush=True)
    print("=" * 72, flush=True)
    for task_id, task in TASKS.items():
        scores = all_scores[task_id]
        avg    = sum(scores) / len(scores)
        detail = "  ".join(f"ep{i+1}={s:.4f}" for i, s in enumerate(scores))
        print(f"  {task.name:<22} [{task.difficulty:<6}]  avg={avg:.4f}   {detail}", flush=True)
    print(flush=True)
    print("Inference complete.", flush=True)


if __name__ == "__main__":
    main()
