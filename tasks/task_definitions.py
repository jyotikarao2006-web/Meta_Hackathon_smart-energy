"""
Task Definitions — Smart Energy Management RL
=============================================
Three tasks in increasing difficulty, each with:
  • A natural-language prompt (what the agent is told to achieve)
  • A grade() function that scores an EpisodeStats → float in [0.0, 1.0]
  • Partial-credit signals so agents can learn from small improvements

Difficulty ladder
-----------------
  Easy   — PeakShaving    : keep daily grid cost below $3  (random agent ~0.2)
  Medium — ComfortFirst   : comfort + cost < $2            (random agent ~0.05)
  Hard   — SolarArbitrage : cost < $1 + sell ≥5 kWh +0 violations  (random ~0.0)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict


# ─────────────────────────────────────────────────────────────
# Lightweight stats container (matches EpisodeStats fields)
# ─────────────────────────────────────────────────────────────

@dataclass
class TaskStats:
    """
    Subset of EpisodeStats needed for task grading.
    Passed into every grade() function.
    """
    total_cost: float          # Total electricity cost for the episode ($)
    comfort_violations: int    # Hours temperature was outside 20-26 °C
    solar_generated: float     # Total solar energy generated (kWh)
    solar_sold_back: float     # kWh sold back to the grid (≥ 0)
    total_reward: float        # Raw environment reward (for diagnostics)
    steps: int                 # Total steps taken (should be 24)


# ─────────────────────────────────────────────────────────────
# Task 1 — PeakShaving  [EASY]
# ─────────────────────────────────────────────────────────────

TASK1_PROMPT = """\
Task: Peak Shaving
Difficulty: Easy
-----------------
Your building has time-of-use pricing that makes electricity 3× more expensive
during peak hours (8–10 am and 5–8 pm). Base load alone costs ~$2.15/day.

Goal: Complete one full 24-hour day with a total grid electricity cost BELOW $6.

Strategy hints:
  • Charge the battery during cheap off-peak hours (night / early morning).
  • Discharge the battery during expensive peak hours to avoid buying grid power.
  • Solar panels supply free power from 6 am–8 pm — use them!

Scoring (0.0 – 1.0):
  • cost ≥ $9.00  →  0.0   (worse than doing nothing)
  • cost = $6.00  →  0.30  (passing threshold)
  • cost = $3.50  →  0.70  (good — smart shaving working)
  • cost = $2.20  →  1.0   (near-optimal — close to base-load-only)
  Comfort violations reduce your score slightly (−0.02 per hour, max −0.15).
"""


def grade_task1(stats: TaskStats) -> float:
    """
    Grade: Peak Shaving (Easy)
    cost ≥9→0, $9→$6: 0→0.30, $6→$3.50: 0.30→0.70, $3.50→$2.20: 0.70→1.0
    Comfort violations: -0.02 each, capped at -0.15.
    """
    cost = stats.total_cost
    if cost >= 9.00:
        cost_score = 0.0
    elif cost >= 6.00:
        cost_score = 0.30 * (9.00 - cost) / (9.00 - 6.00)
    elif cost >= 3.50:
        cost_score = 0.30 + 0.40 * (6.00 - cost) / (6.00 - 3.50)
    elif cost >= 2.20:
        cost_score = 0.70 + 0.30 * (3.50 - cost) / (3.50 - 2.20)
    else:
        cost_score = 1.0
    comfort_penalty = min(0.15, stats.comfort_violations * 0.02)
    return round(max(0.0, min(1.0, cost_score - comfort_penalty)), 4)


# ─────────────────────────────────────────────────────────────
# Task 2 — ComfortFirst  [MEDIUM]
# ─────────────────────────────────────────────────────────────

TASK2_PROMPT = """\
Task: Comfort First
Difficulty: Medium
------------------
Occupants demand a comfortable temperature (20–26 °C) at ALL times.
At the same time, management wants costs kept below $2 for the day.

Goal: Complete 24 hours with ZERO comfort violations AND total cost < $2.

Strategy hints:
  • HVAC is expensive — run it efficiently, not on full blast.
  • Pre-cool or pre-heat the building during cheap off-peak hours so
    you don't need as much HVAC during expensive peak hours.
  • Use the battery to power HVAC during peak hours.

Scoring (0.0 – 1.0):
  • comfort_violations > 0  →  score capped at 0.50, −0.05 per violation
  • violations == 0, cost ≥ $8  →  0.30  (bare pass: comfort held but costly)
  • violations == 0, cost  $8→$4  →  0.30 → 0.65  (real improvement band)
  • violations == 0, cost  $4→$2.20  →  0.65 → 1.0  (excellent: near base-load)
"""


def grade_task2(stats: TaskStats) -> float:
    """
    Grade: Comfort First (Medium)

    Real cost ranges (seed=42, 5 eps):
      random:      ~$8.27   rule-based:  ~$4.32
      conservative:~$4.03   base-load:   ~$2.15

    With comfort violations → capped at 0.50, −0.05 each.
    No violations → cost scoring:
      cost ≥ $8.00  →  0.30  (comfort held, costs still high)
      $8→$4         →  linear 0.30 → 0.65
      $4→$2.20      →  linear 0.65 → 1.00
      cost ≤ $2.20  →  1.00
    """
    violations = stats.comfort_violations
    if violations > 0:
        return round(max(0.0, 0.50 - violations * 0.05), 4)

    cost = stats.total_cost
    if cost >= 8.00:
        cost_score = 0.30
    elif cost >= 4.00:
        cost_score = 0.30 + 0.35 * (8.00 - cost) / (8.00 - 4.00)
    elif cost >= 2.20:
        cost_score = 0.65 + 0.35 * (4.00 - cost) / (4.00 - 2.20)
    else:
        cost_score = 1.00

    return round(min(1.0, cost_score), 4)


# ─────────────────────────────────────────────────────────────
# Task 3 — SolarArbitrage  [HARD]
# ─────────────────────────────────────────────────────────────

TASK3_PROMPT = """\
Task: Solar Arbitrage
Difficulty: Hard
----------------
Master the art of solar energy trading. Store excess solar in the battery
during the day, sell it back during peak pricing, keep the building
comfortable, and drive your net grid cost close to zero (or negative!).

Goals (all three required for a top score):
  1. Total grid cost ≤ $3.00  (base-load ~$2.15; perfect = near zero)
  2. Net solar energy sold back to grid ≥ 5 kWh  (possible: ~9 kWh max)
  3. Zero comfort violations (temperature 20–26 °C throughout)

Strategy hints:
  • Charge the battery aggressively with solar midday (11 am–2 pm).
  • Discharge battery during peak hours (5–8 pm, $0.30/kWh) to avoid buying.
  • Sell excess solar directly when battery is full.
  • Pre-condition the building temperature before peak hours.

Scoring (0.0 – 1.0) — three equal components:
    A) Cost        : 0→0.33 as cost drops $9→$2.20
    B) Sold-back   : 0→0.34 as sold_back grows 0→9 kWh
    C) Comfort     : 0.33 if violations==0, else max(0, 0.33−violations×0.04)
  Perfect score: cost ≤ $2.20 + sold_back ≥ 9 kWh + 0 violations.
"""


def grade_task3(stats: TaskStats) -> float:
    """
    Grade: Solar Arbitrage (Hard)

    Real ranges (seed=42): cost $4–8, sold_back 1–7 kWh, violations ~0.

    A) cost_component (0–0.33)
         cost ≥ $9.00  →  0.0
         $9→$2.20      →  linear 0 → 0.33

    B) sellback_component (0–0.34)
         sold_back → 9 kWh max → linear 0 → 0.34

    C) comfort_component (0–0.33)
         violations==0 → 0.33
         else          → max(0, 0.33 − violations × 0.04)
    """
    cost = stats.total_cost
    if cost >= 9.00:
        cost_comp = 0.0
    elif cost <= 2.20:
        cost_comp = 0.33
    else:
        cost_comp = 0.33 * (9.00 - cost) / (9.00 - 2.20)

    sold = min(stats.solar_sold_back, 9.0)
    sellback_comp = 0.34 * (sold / 9.0)

    violations = stats.comfort_violations
    comfort_comp = 0.33 if violations == 0 else max(0.0, 0.33 - violations * 0.04)

    return round(min(1.0, max(0.0, cost_comp + sellback_comp + comfort_comp)), 4)


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    """A registered evaluation task."""
    id: str
    name: str
    difficulty: str          # "easy" | "medium" | "hard"
    prompt: str
    grade: Callable[[TaskStats], float]

    def __str__(self) -> str:
        return f"Task(id={self.id!r}, difficulty={self.difficulty!r}, name={self.name!r})"


TASKS: Dict[str, Task] = {
    "task1_peak_shaving": Task(
        id="task1_peak_shaving",
        name="Peak Shaving",
        difficulty="easy",
        prompt=TASK1_PROMPT,
        grade=grade_task1,
    ),
    "task2_comfort_first": Task(
        id="task2_comfort_first",
        name="Comfort First",
        difficulty="medium",
        prompt=TASK2_PROMPT,
        grade=grade_task2,
    ),
    "task3_solar_arbitrage": Task(
        id="task3_solar_arbitrage",
        name="Solar Arbitrage",
        difficulty="hard",
        prompt=TASK3_PROMPT,
        grade=grade_task3,
    ),
}


def get_task(task_id: str) -> Task:
    """Retrieve a task by its ID. Raises KeyError for unknown IDs."""
    if task_id not in TASKS:
        available = ", ".join(TASKS.keys())
        raise KeyError(f"Unknown task {task_id!r}. Available: {available}")
    return TASKS[task_id]
