"""
validate.py — Smart Energy RL v2.0
=====================================
Smoke-test suite that:
  1. Imports the environment and runs a full 24-step episode
  2. Checks all OpenEnv API endpoints (reset, step, state, health, tasks)
  3. Validates task graders produce floats in [0, 1]
  4. Prints a pass/fail summary

Run locally:
    python validate.py

Run against a deployed server:
    BASE_URL=http://localhost:7860 python validate.py
"""

import os
import sys
import json
import random
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_URL = os.environ.get("BASE_URL", "")  # empty = test Python directly

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    results.append(ok)

# ─── 1. Direct Python import ───────────────────────────────────
print("\n── 1. Environment import & episode ───────────────────────")
try:
    from env.energy_env import SmartEnergyEnv, PEAK_HOURS
    from models.schemas import EnergyAction
    from tasks.task_definitions import TASKS, TaskStats

    random.seed(42)
    env = SmartEnergyEnv()
    state = env.reset()
    check("Environment import", True)
    check("reset() returns state", state is not None)

    rewards = []
    for _ in range(24):
        a = EnergyAction.random()
        res = env.step(a)
        rewards.append(res.reward)
    check("24-step episode completes", res.done, f"done={res.done}")
    check("Reward is float", isinstance(res.reward, float))
    check("State has battery_health", hasattr(state, "battery_health") or "battery_health" in (state.model_dump() if hasattr(state,"model_dump") else {}))
    check("State has zone_temps", hasattr(state, "zone_temps"))
    check("Solar sold back tracked", env.total_solar_sold_back >= 0)

except Exception as e:
    check("Environment import", False, str(e))
    traceback.print_exc()

# ─── 2. Task graders ──────────────────────────────────────────
print("\n── 2. Task grader validation ──────────────────────────────")
try:
    for task_id, task in TASKS.items():
        for cost, violations, sold_back in [
            (8.0,  5, 0.5),
            (3.5,  0, 3.0),
            (1.5,  0, 8.0),
            (15.0, 20, 0.0),
        ]:
            stats = TaskStats(
                total_cost=cost,
                comfort_violations=violations,
                solar_generated=12.0,
                solar_sold_back=sold_back,
                total_reward=-5.0,
                steps=24,
            )
            score = task.grade(stats)
            ok = isinstance(score, float) and 0.0 <= score <= 1.0
            check(f"{task.name} grade({cost=:.1f},{violations=},{sold_back=:.1f})", ok, f"score={score:.4f}")
except Exception as e:
    check("Task graders", False, str(e))
    traceback.print_exc()

# ─── 3. HTTP API (if server is running) ──────────────────────
if BASE_URL:
    print(f"\n── 3. HTTP API — {BASE_URL} ────────────────────────────")
    try:
        import urllib.request
        import urllib.error

        def api(method, path, body=None):
            url = BASE_URL.rstrip("/") + path
            data = json.dumps(body).encode() if body is not None else b"{}"
            req = urllib.request.Request(url, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())

        code, _ = api("GET", "/health")
        check("GET /health → 200", code == 200)

        code, body = api("POST", "/reset")
        check("POST /reset → 200", code == 200)
        check("/reset returns state", "state" in body)

        code, body = api("GET", "/state")
        check("GET /state → 200", code == 200)

        code, body = api("POST", "/step", {"action": {"hvac_setpoint": 0.5, "battery_action": 0.1}})
        check("POST /step → 200", code == 200)
        check("/step returns reward", "reward" in body)
        check("/step reward is float", isinstance(body.get("reward"), (int, float)))

        code, body = api("GET", "/tasks")
        check("GET /tasks → 200", code == 200)
        check("/tasks returns list", isinstance(body, list) and len(body) == 3)

        # Ping test: 10 sequential steps < 5s total
        t0 = time.time()
        for _ in range(10):
            api("POST", "/step", {"action": {"hvac_setpoint": 0.3, "battery_action": 0.0}})
        elapsed = time.time() - t0
        check("10 steps in < 5s", elapsed < 5.0, f"{elapsed:.2f}s")

    except Exception as e:
        check("HTTP API", False, str(e))
else:
    print("\n── 3. HTTP API — SKIPPED (set BASE_URL to test a live server) ─")

# ─── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 52)
passed = sum(results)
total  = len(results)
print(f"  {passed}/{total} checks passed", "🎉" if passed == total else "⚠️")
print("=" * 52)
sys.exit(0 if passed == total else 1)
