"""
Smart Energy Management RL Environment
=======================================
OpenEnv-compliant environment for training agents to optimise building
energy use across three graded tasks (Easy → Medium → Hard).

Quick start
-----------
    from env.energy_env import SmartEnergyEnv
    from models.schemas import EnergyAction
    from tasks.task_definitions import TASKS, get_task

    env = SmartEnergyEnv()
    state = env.reset()
    result = env.step(EnergyAction(hvac_setpoint=0.5, battery_action=0.2))
"""

from env.energy_env import SmartEnergyEnv
from models.schemas import EnergyAction, EnergyState, StepResult, EpisodeStats
from tasks.task_definitions import TASKS, get_task, TaskStats

__version__ = "1.1.0"
__all__ = [
    "SmartEnergyEnv",
    "EnergyAction",
    "EnergyState",
    "StepResult",
    "EpisodeStats",
    "TASKS",
    "get_task",
    "TaskStats",
]
