"""
Data Models / Schemas — v2.0
============================
Pydantic-based typed models for strict validation and clean JSON serialization.
All OpenEnv endpoints use these models for request/response bodies.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC = True
except ImportError:
    # Fallback to dataclass if pydantic not installed
    from dataclasses import dataclass, field, asdict
    PYDANTIC = False

import random

if PYDANTIC:

    class EnergyState(BaseModel):
        """Complete observation from the environment — what the agent sees."""
        # Time
        hour: int = Field(..., ge=0, le=24,  description="Hour of day (0-23; 24 = episode end)")
        day:  int = Field(..., ge=1,          description="Episode day number")

        # Temperature
        indoor_temp:  float = Field(..., description="Office zone temperature (°C)")
        outdoor_temp: float = Field(..., description="Outdoor ambient temperature (°C)")

        # Battery
        battery_level:    float = Field(..., ge=0.0, description="Current charge (kWh)")
        battery_capacity: float = Field(..., gt=0.0, description="Effective capacity after degradation (kWh)")
        battery_health:   float = Field(1.0, ge=0.0, le=1.0, description="Health fraction (1.0=new, 0.8=degraded)")

        # Energy
        solar_power:  float = Field(..., ge=0.0, description="Current solar output (kW)")
        grid_price:   float = Field(..., gt=0.0, description="Real-time grid price ($/kWh)")
        is_peak_hour: bool  = Field(..., description="True if current hour is high-price peak")

        # Episode accumulators
        total_cost:         float = Field(..., description="Cumulative grid cost today ($)")
        comfort_violations: int   = Field(..., ge=0, description="Zone-hours outside 20-26 °C")

        # Extended observations (differentiation features)
        zone_temps: Dict[str, float] = Field(default_factory=dict, description="Per-zone temperatures {office, server_room, lobby}")
        humidity:   float = Field(50.0, description="Outdoor relative humidity (%)")
        wind_speed: float = Field(1.0,  description="Wind speed (m/s)")

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()

        def to_vector(self) -> list:
            """7-feature normalised vector for RL algorithms."""
            return [
                self.hour / 24.0,
                self.indoor_temp / 40.0,
                self.outdoor_temp / 40.0,
                self.battery_level / max(self.battery_capacity, 1e-6),
                self.solar_power / 4.0,
                self.grid_price / 0.35,
                float(self.is_peak_hour),
            ]

        @property
        def battery_percentage(self) -> float:
            return (self.battery_level / max(self.battery_capacity, 1e-6)) * 100

        @property
        def temp_comfort_status(self) -> str:
            if 20 <= self.indoor_temp <= 26:
                return "✅ Comfortable"
            return "🥶 Too Cold" if self.indoor_temp < 20 else "🔥 Too Hot"


    class EnergyAction(BaseModel):
        """Action submitted by the agent each timestep."""
        hvac_setpoint:  float = Field(0.5, ge=0.0, le=1.0,
                                      description="HVAC power fraction (0=off, 1=full 3.5 kW)")
        battery_action: float = Field(0.0, ge=-1.0, le=1.0,
                                      description="Battery: -1=max discharge, 0=hold, +1=max charge")

        @field_validator("hvac_setpoint")
        @classmethod
        def clamp_hvac(cls, v):
            return max(0.0, min(1.0, v))

        @field_validator("battery_action")
        @classmethod
        def clamp_battery(cls, v):
            return max(-1.0, min(1.0, v))

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()

        @classmethod
        def random(cls) -> "EnergyAction":
            return cls(hvac_setpoint=random.random(), battery_action=random.uniform(-1, 1))

        @classmethod
        def from_vector(cls, vector: list) -> "EnergyAction":
            return cls(hvac_setpoint=float(vector[0]), battery_action=float(vector[1]))


    class StepResult(BaseModel):
        """Response from env.step()."""
        state:  EnergyState
        reward: float = Field(..., description="Step reward (dense, multi-factor)")
        done:   bool  = Field(..., description="True when 24-hour episode completes")
        info:   Dict[str, Any] = Field(default_factory=dict, description="Diagnostic info dict")

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()


    class ResetResponse(BaseModel):
        """Response body for POST /reset."""
        state: EnergyState
        message: str = "Environment reset. New 24-hour episode started."


    class StepRequest(BaseModel):
        """Request body for POST /step."""
        action: EnergyAction


    class EpisodeStats(BaseModel):
        """Summary statistics for a completed episode."""
        total_reward:        float
        total_cost:          float
        comfort_violations:  int
        solar_generated:     float
        avg_battery_level:   float = 0.0
        solar_sold_back:     float = 0.0
        steps:               int

        @property
        def avg_reward_per_step(self) -> float:
            return self.total_reward / max(1, self.steps)

else:
    # ── dataclass fallback (no pydantic) ──────────────────────
    @dataclass
    class EnergyState:
        hour: int; day: int
        indoor_temp: float; outdoor_temp: float
        battery_level: float; battery_capacity: float; battery_health: float = 1.0
        solar_power: float = 0.0; grid_price: float = 0.10; is_peak_hour: bool = False
        total_cost: float = 0.0; comfort_violations: int = 0
        zone_temps: dict = field(default_factory=dict)
        humidity: float = 50.0; wind_speed: float = 1.0

        def to_dict(self): return asdict(self)
        def to_vector(self):
            return [self.hour/24, self.indoor_temp/40, self.outdoor_temp/40,
                    self.battery_level/max(self.battery_capacity,1e-6),
                    self.solar_power/4, self.grid_price/0.35, float(self.is_peak_hour)]
        @property
        def battery_percentage(self): return self.battery_level/max(self.battery_capacity,1e-6)*100
        @property
        def temp_comfort_status(self): return "✅" if 20<=self.indoor_temp<=26 else ("🥶" if self.indoor_temp<20 else "🔥")

    @dataclass
    class EnergyAction:
        hvac_setpoint: float = 0.5; battery_action: float = 0.0
        def __post_init__(self):
            self.hvac_setpoint  = max(0.,min(1.,self.hvac_setpoint))
            self.battery_action = max(-1.,min(1.,self.battery_action))
        def to_dict(self): return asdict(self)
        @classmethod
        def random(cls): return cls(random.random(), random.uniform(-1,1))
        @classmethod
        def from_vector(cls, v): return cls(float(v[0]), float(v[1]))

    @dataclass
    class StepResult:
        state: EnergyState; reward: float; done: bool; info: dict = field(default_factory=dict)
        def to_dict(self): return {"state":asdict(self.state),"reward":self.reward,"done":self.done,"info":self.info}

    @dataclass
    class EpisodeStats:
        total_reward: float; total_cost: float; comfort_violations: int
        solar_generated: float; avg_battery_level: float = 0.0
        solar_sold_back: float = 0.0; steps: int = 24
