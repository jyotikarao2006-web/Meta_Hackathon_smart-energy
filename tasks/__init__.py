"""
OpenEnv Tasks — Smart Energy Management
========================================
Three graded tasks with agent-graded scoring (0.0 – 1.0).

  Task 1 – PeakShaving      (Easy):   Survive 24 h without grid cost exceeding $3
  Task 2 – ComfortFirst     (Medium): Zero comfort violations AND cost < $2
  Task 3 – SolarArbitrage   (Hard):   Sell back ≥ 5 kWh AND cost < $1 AND 0 violations

Each task exposes:
  • task_prompt  : plain-English description shown to the agent
  • grade(stats) : returns float in [0.0, 1.0] with partial-credit signals
"""

from tasks.task_definitions import TASKS, get_task

__all__ = ["TASKS", "get_task"]
