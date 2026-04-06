"""
Smart Energy Management - RL Environment (v2.0 — Competition Grade)
====================================================================
Upgrades over v1:
  • Dynamic electricity pricing  (real-time price signal, not just binary peak/off-peak)
  • Battery degradation model     (cycle counting, capacity fade, health %)
  • Weather simulation            (temperature, cloud cover, humidity)
  • Multi-room HVAC control       (3 zones: office, server room, lobby)
  • Dense, multi-factor reward    (partial-progress signals throughout)

OpenEnv interface: reset() → EnergyState | step(EnergyAction) → StepResult | state() → EnergyState
"""

import random
import math
from typing import Tuple, Dict, Any
from models.schemas import EnergyState, EnergyAction, StepResult


# ──────────────────────────────────────────────────────────────────
# Dynamic pricing table ($/kWh per hour — realistic grid simulation)
# ──────────────────────────────────────────────────────────────────
_BASE_PRICE_CURVE = [
    0.08, 0.07, 0.07, 0.08,   # 00-03  deep night (cheapest)
    0.09, 0.10, 0.12, 0.15,   # 04-07  pre-morning ramp
    0.28, 0.32, 0.30, 0.22,   # 08-11  morning peak
    0.18, 0.16, 0.15, 0.16,   # 12-15  afternoon shoulder
    0.20, 0.28, 0.35, 0.30,   # 16-19  evening peak (highest)
    0.22, 0.16, 0.12, 0.09,   # 20-23  night ramp-down
]
PEAK_HOURS = [h for h, p in enumerate(_BASE_PRICE_CURVE) if p >= 0.25]


class WeatherSimulator:
    """Realistic daily weather with stochastic perturbations."""

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        roll = random.random()
        if roll < 0.5:
            self.cloud_base = random.uniform(0.0, 0.15)
        elif roll < 0.80:
            self.cloud_base = random.uniform(0.15, 0.45)
        else:
            self.cloud_base = random.uniform(0.45, 0.80)
        self.base_temp  = random.uniform(22.0, 32.0)
        self.temp_amp   = random.uniform(5.0, 9.0)
        self.wind_speed = random.uniform(0.5, 6.0)

    def outdoor_temp(self, hour: int) -> float:
        phase = math.sin(math.pi * (hour - 6) / 12)
        return round(self.base_temp + self.temp_amp * phase + random.uniform(-0.8, 0.8), 1)

    def solar_irradiance(self, hour: int) -> float:
        if hour < 6 or hour >= 20:
            return 0.0
        raw = 4.0 * math.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
        cloud = max(0.0, min(0.95, self.cloud_base + random.uniform(-0.05, 0.10)))
        return round(raw * (1.0 - cloud), 2)

    def humidity(self, hour: int) -> float:
        base = 55.0 + 20.0 * math.cos(math.pi * (hour - 6) / 12)
        return round(min(95.0, max(20.0, base + random.uniform(-5, 5))), 1)

    def wind(self) -> float:
        return round(max(0.0, self.wind_speed + random.uniform(-0.5, 0.5)), 1)


class BatteryModel:
    """Li-ion battery with degradation tracking (cycle counting, capacity fade)."""

    FULL_CAPACITY         = 10.0
    MAX_RATE              = 2.0
    CHARGE_EFF            = 0.95
    DISCHARGE_EFF         = 0.97
    DEGRADATION_PER_CYCLE = 0.00003

    def __init__(self):
        self.level  = random.uniform(2.0, 5.0)
        self.cycles = 0.0
        self.health = 1.0

    @property
    def capacity(self) -> float:
        return self.FULL_CAPACITY * self.health

    def apply_action(self, battery_action: float) -> float:
        """Returns grid_draw (positive=consuming, negative=supplying)."""
        requested = battery_action * self.MAX_RATE
        if requested > 0:
            headroom = self.capacity - self.level
            actual = min(requested, headroom, self.MAX_RATE)
            self.level += actual * self.CHARGE_EFF
            grid_draw = actual
        else:
            actual = max(requested, -self.level, -self.MAX_RATE)
            self.level += actual / self.DISCHARGE_EFF
            grid_draw = actual
        self.level = max(0.0, min(self.capacity, self.level))
        half_cycle = abs(actual) / (self.capacity + 1e-9)
        self.cycles += half_cycle * 0.5
        self.health = max(0.80, 1.0 - self.cycles * self.DEGRADATION_PER_CYCLE)
        return round(grid_draw, 4)

    def reset_episode(self):
        self.level = random.uniform(2.0, 5.0)


class MultiRoomHVAC:
    """Three-zone HVAC: office, server_room, lobby."""

    ZONES = {
        "office":      {"mass": 8.0, "setpoint": 22.0, "max_kw": 2.0, "base_gain_kw": 0.0},
        "server_room": {"mass": 3.0, "setpoint": 20.0, "max_kw": 1.0, "base_gain_kw": 1.2},
        "lobby":       {"mass": 2.0, "setpoint": 23.0, "max_kw": 0.5, "base_gain_kw": 0.0},
    }
    COMFORT_MIN = 20.0
    COMFORT_MAX = 26.0

    def __init__(self):
        self.temps = {z: random.uniform(20.0, 24.0) for z in self.ZONES}

    def step(self, hvac_setpoints: Dict[str, float], outdoor_temp: float):
        total_kw = 0.0
        violations = 0
        for zone, cfg in self.ZONES.items():
            sp = max(0.0, min(1.0, hvac_setpoints.get(zone, 0.5)))
            total_kw += sp * cfg["max_kw"]
            T = self.temps[zone]
            T += (outdoor_temp - T) * 0.04
            T += (cfg["setpoint"] - T) * sp * 0.5
            T += cfg["base_gain_kw"] * 0.3
            T = max(10.0, min(45.0, T))
            self.temps[zone] = round(T, 2)
            if T < self.COMFORT_MIN or T > self.COMFORT_MAX:
                violations += 1
        mean_temp = round(sum(self.temps.values()) / len(self.temps), 2)
        return round(total_kw, 3), violations, mean_temp

    def reset_episode(self):
        self.temps = {z: random.uniform(20.0, 24.0) for z in self.ZONES}

    @property
    def zone_temps(self) -> Dict[str, float]:
        return dict(self.temps)


class SmartEnergyEnv:
    """
    Smart Energy Management Environment — v2.0 (Competition Grade).

    Observation space: hour, indoor_temp, outdoor_temp, battery_level,
        battery_capacity, solar_power, grid_price, is_peak_hour,
        total_cost, comfort_violations, battery_health

    Action space (continuous):
        hvac_setpoint   [0, 1]
        battery_action  [-1, 1]

    Reward: dense, multi-factor, partial-progress
    """

    BASE_LOAD_KW   = 1.5
    SOLAR_SELLBACK = 0.05

    def __init__(self):
        self.battery = BatteryModel()
        self.hvac    = MultiRoomHVAC()
        self.weather = WeatherSimulator()
        self.hour    = 0
        self.day     = 0
        self.max_hours = 24

        self.total_cost               = 0.0
        self.total_comfort_violations = 0
        self.total_solar_generated    = 0.0
        self.total_solar_sold_back    = 0.0

        self._current_state = None
        self.reset()

    # ── OpenEnv interface ─────────────────────────────────────

    def reset(self) -> EnergyState:
        self.hour  = 0
        self.day  += 1
        self.weather = WeatherSimulator()
        self.battery.reset_episode()
        self.hvac.reset_episode()
        self.total_cost               = 0.0
        self.total_comfort_violations = 0
        self.total_solar_generated    = 0.0
        self.total_solar_sold_back    = 0.0
        self._current_state = self._build_state()
        return self._current_state

    def step(self, action: EnergyAction) -> StepResult:
        solar_kw     = self.weather.solar_irradiance(self.hour)
        outdoor_temp = self.weather.outdoor_temp(self.hour)
        grid_price   = self._dynamic_grid_price(self.hour)
        is_peak      = self.hour in PEAK_HOURS

        self.total_solar_generated += solar_kw

        bat_act           = max(-1.0, min(1.0, action.battery_action))
        grid_draw_battery = self.battery.apply_action(bat_act)

        hvac_sp        = max(0.0, min(1.0, action.hvac_setpoint))
        zone_setpoints = {z: hvac_sp for z in MultiRoomHVAC.ZONES}
        hvac_kw, zone_violations, mean_indoor_temp = self.hvac.step(zone_setpoints, outdoor_temp)
        self.total_comfort_violations += zone_violations

        total_load    = hvac_kw + self.BASE_LOAD_KW + max(0.0, grid_draw_battery)
        battery_supply = max(0.0, -grid_draw_battery)
        net_from_grid  = total_load - solar_kw - battery_supply

        if net_from_grid > 0:
            hourly_cost = net_from_grid * grid_price
        else:
            hourly_cost = net_from_grid * self.SOLAR_SELLBACK
            self.total_solar_sold_back += abs(net_from_grid)

        self.total_cost += hourly_cost

        reward = self._calculate_reward(
            hourly_cost=hourly_cost,
            zone_violations=zone_violations,
            solar_kw=solar_kw,
            bat_grid_draw=grid_draw_battery,
            is_peak=is_peak,
            net_from_grid=net_from_grid,
        )

        self.hour += 1
        done = self.hour >= self.max_hours
        self._current_state = self._build_state()

        info = {
            "hourly_cost":             round(hourly_cost, 4),
            "solar_power_kw":          round(solar_kw, 2),
            "grid_price":              round(grid_price, 4),
            "is_peak_hour":            is_peak,
            "battery_level":           round(self.battery.level, 2),
            "battery_health":          round(self.battery.health, 4),
            "hvac_kw":                 round(hvac_kw, 2),
            "zone_temps":              self.hvac.zone_temps,
            "zone_violations":         zone_violations,
            "net_from_grid_kw":        round(net_from_grid, 3),
            "total_cost_so_far":       round(self.total_cost, 4),
            "total_comfort_violations":self.total_comfort_violations,
            "total_solar_sold_back_kwh":round(self.total_solar_sold_back, 3),
            "outdoor_temp":            outdoor_temp,
            "humidity":                self.weather.humidity(self.hour),
            "wind_speed":              self.weather.wind(),
        }

        return StepResult(
            state  = self._current_state,
            reward = round(reward, 4),
            done   = done,
            info   = info,
        )

    def state(self) -> EnergyState:
        return self._current_state

    # ── Private ───────────────────────────────────────────────

    def _dynamic_grid_price(self, hour: int) -> float:
        base = _BASE_PRICE_CURVE[hour % 24]
        return round(max(0.05, base + random.uniform(-0.015, 0.015)), 4)

    def _calculate_reward(self, hourly_cost, zone_violations, solar_kw,
                          bat_grid_draw, is_peak, net_from_grid) -> float:
        r = 0.0

        # 1. Cost (dense, continuous)
        r -= hourly_cost * 10.0

        # 2. Comfort: partial credit per zone
        ok_zones = len(MultiRoomHVAC.ZONES) - zone_violations
        r += (ok_zones / len(MultiRoomHVAC.ZONES)) * 2.0
        for T in self.hvac.zone_temps.values():
            if T < MultiRoomHVAC.COMFORT_MIN:
                r -= (MultiRoomHVAC.COMFORT_MIN - T) * 0.5
            elif T > MultiRoomHVAC.COMFORT_MAX:
                r -= (T - MultiRoomHVAC.COMFORT_MAX) * 0.5

        # 3. Solar utilisation
        solar_used = max(0.0, solar_kw - max(0.0, -net_from_grid))
        r += solar_used * 0.4

        # 4. Battery strategy
        bat_pct = self.battery.level / (self.battery.capacity + 1e-9)
        if not is_peak and bat_grid_draw > 0 and solar_kw > 1.0:
            r += 0.3   # charging with solar
        if is_peak and bat_grid_draw < 0:
            r += 0.4   # discharging at peak
        if 0.25 <= bat_pct <= 0.75:
            r += 0.2   # healthy SoC
        r += self.battery.health * 0.05  # health preservation

        # 5. Grid export bonus at peak
        if net_from_grid < 0 and is_peak:
            r += abs(net_from_grid) * 0.5

        return r

    def _build_state(self) -> EnergyState:
        h = self.hour
        return EnergyState(
            hour               = h,
            day                = self.day,
            indoor_temp        = self.hvac.zone_temps.get("office", 22.0),
            outdoor_temp       = self.weather.outdoor_temp(h),
            battery_level      = round(self.battery.level, 2),
            battery_capacity   = round(self.battery.capacity, 2),
            solar_power        = self.weather.solar_irradiance(h),
            grid_price         = self._dynamic_grid_price(h),
            is_peak_hour       = h in PEAK_HOURS,
            total_cost         = round(self.total_cost, 4),
            comfort_violations = self.total_comfort_violations,
            battery_health     = round(self.battery.health, 4),
            zone_temps         = self.hvac.zone_temps,
            humidity           = self.weather.humidity(h),
            wind_speed         = self.weather.wind(),
        )
