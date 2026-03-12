"""
OpenServoSim - OP3 Running Environment

Subclasses the proven Op3Joystick environment to add running-specific rewards:
1. Flight phase reward (encourages both feet off the ground)
2. Relaxed z-axis velocity penalty (allows bouncing)
3. Adjusted action_rate penalty (allows high-frequency steps)
"""

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src.locomotion.op3 import joystick
from ml_collections import config_dict


def default_running_config() -> config_dict.ConfigDict:
    cfg = joystick.default_config()

    # Running commands
    cfg.lin_vel_x = [1.5, 4.0]     # High forward speed
    cfg.lin_vel_y = [-0.1, 0.1]    # Minimal lateral
    cfg.ang_vel_yaw = [-0.2, 0.2]  # Minimal turning

    # Running rewards
    cfg.reward_config.scales.tracking_lin_vel = 15.0   # Must run forward
    cfg.reward_config.scales.tracking_ang_vel = 5.0
    cfg.reward_config.scales.orientation = -5.0        # Stay upright, but allow slight lean
    cfg.reward_config.scales.lin_vel_z = 0.0           # REMOVED: Bouncing is required for running!
    cfg.reward_config.scales.ang_vel_xy = -0.5         # Keep torso stable (less wobble)
    cfg.reward_config.scales.action_rate = -0.005      # RELAXED: Allow faster leg movements
    cfg.reward_config.scales.torques = -0.0002         # Energy efficiency
    cfg.reward_config.scales.termination = -1.0        # Don't fall
    cfg.reward_config.scales.feet_clearance = 1.0      # HIGHER: Must lift feet to run
    cfg.reward_config.scales.feet_slip = -0.1          # No slipping
    cfg.reward_config.scales.flight = 2.0              # NEW: Reward for having both feet off ground
    cfg.reward_config.scales.energy = -0.0001
    
    # Increase action scale to allow larger strides
    cfg.action_scale = 0.5 

    return cfg


class Op3RunningEnv(joystick.Joystick):
    """OP3 environment modified for running (flight phase)."""

    def __init__(self, config=None, config_overrides=None):
        if config is None:
            config = default_running_config()
        super().__init__(config, config_overrides)

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict,
        metrics: dict,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        """Override to add running-specific rewards."""
        
        # Base joystick rewards
        rewards = super()._get_reward(data, action, info, metrics, done)

        # Remove redundant/counter-productive original rewards
        for key in ["lin_vel_z", "tracking_lin_vel"]:
            if key in rewards:
                rewards[key] = jp.zeros(())

        # --- Calculate Forward Velocity Reward (LINEAR) ---
        # Instead of exp(-error), we just directly reward moving forward
        # This fixes the vanishing gradient problem at high command speeds.
        forward_vel = self.get_global_linvel(data)[0]
        rewards["forward_velocity"] = jp.clip(forward_vel, 0.0, 5.0)
        
        # --- Calculate Flight Phase Reward ---
        left_feet_contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._left_feet_floor_found_sensor
        ])
        right_feet_contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._right_feet_floor_found_sensor
        ])
        
        is_left_touching = jp.any(left_feet_contact)
        is_right_touching = jp.any(right_feet_contact)
        is_flight = ~(is_left_touching | is_right_touching)
        moving_fast = forward_vel > 0.5
        rewards["flight"] = jp.where(is_flight & moving_fast, 1.0, 0.0)

        # --- Survival Bonus ---
        # Constant reward per step to offset small stability penalties
        # and encourage the robot to stay alive while learning to run.
        rewards["survival"] = 1.0 - done

        return rewards
