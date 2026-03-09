"""
=============================================================================
  UVC Walking Controller — Unified Balance + Gait for OP3
=============================================================================

  Combines Dr. Guero's UVC (Upper Body Vertical Control) balance algorithm
  with ROBOTIS-style sinusoidal walking gait.

  Key design (from Dr. Guero's core.cpp):
    - Walking engine runs normally to produce base joint angles
    - UVC reads IMU pitch/roll and computes balance corrections
    - Corrections are applied as additive hip/ankle offsets
    - This avoids re-implementing the gait and ensures consistency

  Reference:
    - UVC: http://ai2001.ifdef.jp/uvc/code_Eng.html
    - Walking: ROBOTIS OP3 op3_walking_module
=============================================================================
"""

import math
import numpy as np
from dataclasses import dataclass

from controllers.robotis_walking import (
    WalkingParam, WalkingEngine, wsin, solve_ik_simple,
    THIGH_LENGTH, CALF_LENGTH, ANKLE_LENGTH, LEG_LENGTH, LEG_SIDE_OFFSET,
)


# ============================================================================
# UVC Parameters
# ============================================================================

@dataclass
class UVCParam:
    """UVC balance tuning parameters."""
    # Response gains (0.25 = conservative, 0.85 = aggressive)
    gain_roll: float = 0.20
    gain_pitch: float = 0.15

    # Dead zone: ignore tilt smaller than this (radians)
    dead_zone: float = 0.05  # ~2.9 degrees (wider to avoid jitter)

    # Integration clamp (radians for additive correction)
    max_correction: float = 0.10  # ~5.7 degrees max correction per joint

    # Integration scale (how much IMU angle maps to joint correction)
    roll_scale: float = 0.15
    pitch_scale: float = 0.10

    # Decay rate per step (multiplicative)
    roll_decay: float = 0.92
    pitch_decay: float = 0.90

    # Warmup time: UVC starts this many seconds after walking begins
    warmup_time: float = 2.0

    # Enable/disable
    enabled: bool = True


# ============================================================================
# Unified UVC + Walking Controller
# ============================================================================

class UVCWalkingEngine:
    """
    Unified controller: ROBOTIS walking gait + UVC balance.

    Architecture:
        1. WalkingEngine.update(dt) produces base joint angles
        2. UVC reads IMU and computes additive corrections
        3. Corrections are applied to hip pitch/roll and ankle pitch/roll
        4. WalkingEngine.apply_balance() adds fast ankle feedback on top

    Usage:
        engine = UVCWalkingEngine()
        engine.set_velocity(forward=0.025)
        engine.start()

        while running:
            angles = engine.update(dt, pitch, roll)
            WalkingEngine.apply_balance(angles, pitch, roll)
            # Apply angles to actuators
    """

    def __init__(self, walk_params: WalkingParam = None, uvc_params: UVCParam = None):
        # Walking engine (handles all gait computation)
        self.walk = WalkingEngine(walk_params)
        self.uvc_p = uvc_params or UVCParam()

        # UVC accumulated corrections (radians, additive to joints)
        self._pitch_corr = 0.0   # Forward/backward correction
        self._roll_corr = 0.0    # Lateral correction
        self._walk_time = 0.0    # Time since walking started

        # IMU calibration
        self._pitch_offset = 0.0
        self._roll_offset = 0.0
        self._cal_samples = []

    def set_velocity(self, forward: float = 0.0, lateral: float = 0.0, turn: float = 0.0):
        """Set walking velocity."""
        self.walk.set_velocity(forward, lateral, turn)

    def start(self):
        """Start walking with UVC balance."""
        self.walk.start()
        self._pitch_corr = 0.0
        self._roll_corr = 0.0
        self._walk_time = 0.0

    def stop(self):
        """Stop walking."""
        self.walk.stop()

    def calibrate_imu(self, pitch: float, roll: float):
        """Accumulate IMU calibration samples."""
        self._cal_samples.append((pitch, roll))
        if len(self._cal_samples) >= 100:
            self._pitch_offset = sum(s[0] for s in self._cal_samples) / len(self._cal_samples)
            self._roll_offset = sum(s[1] for s in self._cal_samples) / len(self._cal_samples)
            self._cal_samples = []

    def update(self, dt: float, pitch: float = 0.0, roll: float = 0.0) -> dict:
        """
        Advance the unified controller by dt seconds.

        Args:
            dt: Time step in seconds
            pitch: Body pitch from IMU (radians, positive = forward lean)
            roll: Body roll from IMU (radians, positive = left tilt)

        Returns: dict of actuator_name -> target_angle (radians)
        """
        if not self.walk.running:
            return self.walk._standing_pose()

        # === Step 1: Walking engine produces base joint angles ===
        angles = self.walk.update(dt)

        # === Step 2: UVC balance corrections ===
        self._walk_time += dt

        if self.uvc_p.enabled and self._walk_time > self.uvc_p.warmup_time:
            # Remove IMU offset
            p_corr = pitch - self._pitch_offset
            r_corr = roll - self._roll_offset

            # Dead zone
            tilt_mag = math.sqrt(p_corr**2 + r_corr**2)
            if tilt_mag > self.uvc_p.dead_zone:
                k1 = (tilt_mag - self.uvc_p.dead_zone) / tilt_mag
                p_corr *= k1
                r_corr *= k1
            else:
                p_corr = 0.0
                r_corr = 0.0

            # Integrate with gain
            self._pitch_corr += self.uvc_p.gain_pitch * p_corr * self.uvc_p.pitch_scale
            self._roll_corr += self.uvc_p.gain_roll * r_corr * self.uvc_p.roll_scale

            # Decay
            self._pitch_corr *= self.uvc_p.pitch_decay
            self._roll_corr *= self.uvc_p.roll_decay

            # Clamp
            mc = self.uvc_p.max_correction
            self._pitch_corr = max(-mc, min(mc, self._pitch_corr))
            self._roll_corr = max(-mc, min(mc, self._roll_corr))

            # === Step 3: Apply UVC corrections to hip and ankle joints ===
            # Pitch correction: lean forward/backward
            # Both hips shift, both ankles compensate
            angles["r_hip_pitch_act"] += self._pitch_corr
            angles["l_hip_pitch_act"] -= self._pitch_corr  # L is negated axis

            # Roll correction: shift weight laterally
            angles["r_hip_roll_act"] += self._roll_corr
            angles["l_hip_roll_act"] += self._roll_corr  # Same axis convention

        return angles

    @property
    def phase_name(self) -> str:
        return self.walk.phase_name

    def get_uvc_info(self) -> dict:
        """Return UVC state for diagnostics."""
        return {
            "pitch_corr": self._pitch_corr,
            "roll_corr": self._roll_corr,
            "uvc_enabled": self.uvc_p.enabled,
        }
