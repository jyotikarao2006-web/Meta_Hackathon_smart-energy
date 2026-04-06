"""
Tests for Smart Energy Environment
====================================
These tests verify our environment works correctly.

Think of tests as "quality checks" - they automatically verify
that all the core functionality behaves as expected.

HOW TO RUN:
    python tests/test_env.py
    
    OR run all tests with:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.energy_env import SmartEnergyEnv
from models.schemas import EnergyAction, EnergyState, StepResult


def test_reset():
    """Test that reset() returns a valid initial state."""
    print("Testing reset()...")
    
    env = SmartEnergyEnv()
    state = env.reset()
    
    # Check it returns the right type
    assert isinstance(state, EnergyState), "reset() should return EnergyState"
    
    # Check starting values make sense
    assert state.hour == 0, "Should start at hour 0"
    assert 0 <= state.battery_level <= 10.0, "Battery should be 0-10 kWh"
    assert 18 <= state.indoor_temp <= 26, "Indoor temp should be reasonable"
    assert state.total_cost == 0.0, "No cost at episode start"
    assert state.comfort_violations == 0, "No violations at episode start"
    
    print("  ✅ reset() PASSED")


def test_step_basic():
    """Test that step() works and returns correct structure."""
    print("Testing step()...")
    
    env = SmartEnergyEnv()
    env.reset()
    
    # Take a simple action
    action = EnergyAction(hvac_setpoint=0.5, battery_action=0.0)
    result = env.step(action)
    
    # Check return type
    assert isinstance(result, StepResult), "step() should return StepResult"
    assert isinstance(result.state, EnergyState), "result.state should be EnergyState"
    assert isinstance(result.reward, float), "reward should be a float"
    assert isinstance(result.done, bool), "done should be a boolean"
    
    # After first step, we're at hour 1 (not done)
    assert result.state.hour == 1, "Should advance to hour 1"
    assert result.done == False, "Should not be done after 1 step"
    
    print("  ✅ step() basic PASSED")


def test_episode_completes():
    """Test that running 24 steps completes the episode."""
    print("Testing full episode (24 steps)...")
    
    env = SmartEnergyEnv()
    env.reset()
    
    done = False
    steps = 0
    result = None
    
    while not done:
        action = EnergyAction.random()
        result = env.step(action)
        done = result.done
        steps += 1
        
        # Safety check: shouldn't take more than 24 steps
        assert steps <= 24, f"Episode should end by step 24, but at step {steps}"
    
    assert steps == 24, f"Episode should take exactly 24 steps, took {steps}"
    assert result.done == True, "Last step should have done=True"
    
    print(f"  ✅ Episode completed in {steps} steps — PASSED")


def test_state_method():
    """Test that state() returns current state without advancing."""
    print("Testing state()...")
    
    env = SmartEnergyEnv()
    initial_state = env.reset()
    
    # Get state without stepping
    current_state = env.state()
    
    assert current_state.hour == initial_state.hour, "state() should match reset() state"
    assert current_state.battery_level == initial_state.battery_level, "Battery should match"
    
    # Take a step and verify state() reflects it
    env.step(EnergyAction(hvac_setpoint=0.5, battery_action=0.0))
    new_state = env.state()
    
    assert new_state.hour == 1, "After step, state() should show hour 1"
    
    print("  ✅ state() PASSED")


def test_action_validation():
    """Test that invalid actions are clamped to valid range."""
    print("Testing action validation...")
    
    # These values are out of range
    action = EnergyAction(hvac_setpoint=5.0, battery_action=-99.0)
    
    # Should be automatically clamped
    assert action.hvac_setpoint == 1.0, "HVAC should be clamped to 1.0"
    assert action.battery_action == -1.0, "Battery should be clamped to -1.0"
    
    print("  ✅ Action validation PASSED")


def test_reward_is_reasonable():
    """Test that rewards are in a reasonable range."""
    print("Testing reward ranges...")
    
    env = SmartEnergyEnv()
    env.reset()
    
    rewards = []
    for _ in range(24):
        action = EnergyAction(hvac_setpoint=0.5, battery_action=0.0)
        result = env.step(action)
        rewards.append(result.reward)
        if result.done:
            break
    
    # Rewards shouldn't be insanely large or small
    for r in rewards:
        assert -50 < r < 50, f"Reward {r} seems unreasonable"
    
    print(f"  Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print("  ✅ Reward ranges PASSED")


def test_state_to_vector():
    """Test state serialization to vector."""
    print("Testing state serialization...")
    
    env = SmartEnergyEnv()
    state = env.reset()
    
    vector = state.to_vector()
    state_dict = state.to_dict()
    
    assert len(vector) == 7, f"Vector should have 7 elements, has {len(vector)}"
    assert all(isinstance(v, float) for v in vector), "All vector elements should be float"
    assert "hour" in state_dict, "Dict should have 'hour' key"
    assert "battery_level" in state_dict, "Dict should have 'battery_level' key"
    
    print(f"  State vector: {[round(v, 2) for v in vector]}")
    print("  ✅ State serialization PASSED")


def test_solar_only_during_day():
    """Test that solar is 0 at night."""
    print("Testing solar day/night cycle...")
    
    env = SmartEnergyEnv()
    env.reset()
    
    # Run for a few hours starting from midnight
    night_solar = env._get_solar_power(3)   # 3am - should be 0
    day_solar = env._get_solar_power(13)    # 1pm - should be positive
    
    assert night_solar == 0.0, "No solar power at 3am"
    assert day_solar > 0.0, "Should have solar power at 1pm"
    
    print(f"  Solar at 3am: {night_solar} kW")
    print(f"  Solar at 1pm: {day_solar:.2f} kW")
    print("  ✅ Solar cycle PASSED")


if __name__ == "__main__":
    print("\n🧪 Running Smart Energy Environment Tests")
    print("=" * 45)
    
    tests = [
        test_reset,
        test_step_basic,
        test_episode_completes,
        test_state_method,
        test_action_validation,
        test_reward_is_reasonable,
        test_state_to_vector,
        test_solar_only_during_day,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*45}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Environment is working correctly.")
    else:
        print("⚠️  Some tests failed. Check output above.")
