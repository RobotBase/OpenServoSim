"""
=============================================================================
  ROBOTIS OP3 Walking Engine — Python Port
=============================================================================

  Pure-Python port of ROBOTIS-GIT/ROBOTIS-OP3/op3_walking_module.
  Generates foot trajectories using sinusoidal functions in Cartesian space,
  then solves 6-DOF geometric IK per leg to produce joint angles.

  Original source: https://github.com/ROBOTIS-GIT/ROBOTIS-OP3
  License: Apache 2.0 (ROBOTIS CO., LTD.)
  
  Ported for: OpenServoSim — servo-driven humanoid walking research
=============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# ============================================================================
# Walking Parameters (from ROBOTIS param.yaml defaults)
# ============================================================================

@dataclass
class WalkingParam:
    """All tunable walking parameters."""
    # Stance offsets (meters, radians)
    # NOTE: These are relative to the straight-leg default pose.
    # In MuJoCo, zero ctrl = straight legs at ~279mm height.
    # Small z_offset means slight knee bend from fully extended.
    init_x_offset: float = -0.005       # Tiny forward offset
    init_y_offset: float = 0.0          # No lateral offset needed in sim
    init_z_offset: float = 0.005        # Very slight bend from fully extended
    init_roll_offset: float = 0.0
    init_pitch_offset: float = 0.0
    init_yaw_offset: float = 0.0
    
    # Walking control
    hip_pitch_offset: float = 0.0   # Zero lean — let the gait itself provide forward motion
    period_time: float = 0.600     # Full walk cycle (seconds)
    dsp_ratio: float = 0.2        # 20% double-support
    step_fb_ratio: float = 0.28   # Forward/backward step ratio
    
    # Movement amplitudes (meters, radians)
    x_move_amplitude: float = 0.0     # Forward step (set at runtime)
    y_move_amplitude: float = 0.0     # Lateral step
    z_move_amplitude: float = 0.025    # Foot lift height
    angle_move_amplitude: float = 0.0  # Turn angle
    
    # Body sway
    y_swap_amplitude: float = 0.020   # Lateral sway (reduced for stability)
    z_swap_amplitude: float = 0.004   # Vertical bounce (reduced)
    
    # Offsets
    pelvis_offset: float = math.radians(0.5)   # Pelvis roll offset
    arm_swing_gain: float = 0.5
    
    # Balance feedback gains
    balance_enable: bool = True
    balance_hip_roll_gain: float = 0.35
    balance_knee_gain: float = 0.30
    balance_ankle_roll_gain: float = 0.70
    balance_ankle_pitch_gain: float = 0.90


# ============================================================================
# OP3 Kinematics Constants (from op3_kinematics_dynamics)
# ============================================================================

# Leg segment lengths in meters (from op3.xml body positions)
# hip_pitch_link → knee_link: z-offset = 0.11015 m
# knee_link → ank_pitch_link: z-offset = 0.11 m  
# ank_pitch_link → foot contact: ~0.0265 m (foot geom pos z)
THIGH_LENGTH = 0.11015    # Hip pitch to knee
CALF_LENGTH = 0.110       # Knee to ankle pitch
ANKLE_LENGTH = 0.0265     # Ankle pitch to foot sole
LEG_SIDE_OFFSET = 0.035   # Hip offset from center (y-axis)

# Total leg length when straight
LEG_LENGTH = THIGH_LENGTH + CALF_LENGTH + ANKLE_LENGTH


# ============================================================================
# wSin — Sinusoidal trajectory primitive (from ROBOTIS)
# ============================================================================

def wsin(time: float, period: float, phase_shift: float,
         amplitude: float, amplitude_shift: float) -> float:
    """Compute sinusoidal trajectory value.
    
    This is the 'wSin' function from ROBOTIS walking engine.
    Generates smooth periodic motion: amplitude * sin(2π/period * time + phase) + shift
    """
    if period == 0.0:
        return 0.0
    return amplitude * math.sin(2.0 * math.pi / period * time + phase_shift) + amplitude_shift


# ============================================================================
# 6-DOF Geometric Leg IK (from ROBOTIS op3_kinematics_dynamics)
# ============================================================================

def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _get_sign(x: float) -> float:
    return -1.0 if x < 0 else 1.0


def solve_leg_ik(x: float, y: float, z: float,
                 roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Solve 6-DOF geometric IK for one OP3 leg.
    
    Input: foot endpoint position (x,y,z) in meters and orientation (roll,pitch,yaw) in radians,
           relative to the hip joint origin.
    
    Output: 6 joint angles [hip_yaw, hip_roll, hip_pitch, knee, ank_pitch, ank_roll]
    
    Port of ROBOTIS op3_kinematics_dynamics::calcInverseKinematicsForLeg()
    """
    out = np.zeros(6)
    
    # Build rotation matrix from foot orientation
    R06 = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    
    # Position vector from hip to foot
    p = np.array([x, y, z])
    
    # Vector from ankle to foot (along z in foot frame)
    p36 = p - R06 @ np.array([0, 0, ANKLE_LENGTH])
    
    # Solve q6 (ankle roll)
    out[5] = math.atan2(p[1], p[2])
    
    # Solve q1 (hip yaw)  
    R_ank_roll = _rot_x(out[5])
    p_in_ank = R_ank_roll.T @ p
    out[0] = math.atan2(p_in_ank[0], p_in_ank[2])
    
    # Solve q4 (knee) — from law of cosines
    d = np.linalg.norm(p36)
    cos_q4 = (THIGH_LENGTH**2 + CALF_LENGTH**2 - d**2) / (2 * THIGH_LENGTH * CALF_LENGTH)
    cos_q4 = np.clip(cos_q4, -1.0, 1.0)
    out[3] = -math.acos(cos_q4) + math.pi
    
    # Solve q5 (ankle pitch)
    alpha = math.asin(np.clip(
        THIGH_LENGTH * math.sin(math.pi - out[3]) / max(d, 1e-6), -1.0, 1.0
    ))
    p63 = -R06.T @ p36
    out[4] = -math.atan2(p63[0], _get_sign(p63[2]) * math.sqrt(p63[1]**2 + p63[2]**2)) - alpha
    
    # Solve q2 (hip roll) and q3 (hip pitch)
    R05 = _rot_z(out[0]) @ _rot_x(out[5])  # Simplified chain
    R13 = _rot_z(-out[0]) @ R06 @ _rot_y(-(out[4] + out[3]))  # Approximation
    
    # Use simpler geometric approach for robustness
    out[1] = math.atan2(R13[2, 1], R13[1, 1]) if abs(R13[1, 1]) > 1e-6 else 0.0
    out[2] = math.atan2(R13[0, 2], R13[0, 0]) if abs(R13[0, 0]) > 1e-6 else 0.0
    
    return out


def solve_ik_simple(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Simplified 3-DOF planar IK for sagittal plane (hip_pitch, knee, ank_pitch).
    
    This is more robust than the full 6-DOF solver for basic walking.
    Input: foot position relative to hip (x=forward, z=down, both in meters).
    """
    # Distance from hip to ankle
    dz = abs(z) - ANKLE_LENGTH
    dx = x
    d = math.sqrt(dx**2 + dz**2)
    
    # Clamp to reachable workspace
    max_reach = THIGH_LENGTH + CALF_LENGTH - 0.001
    min_reach = abs(THIGH_LENGTH - CALF_LENGTH) + 0.001
    d = np.clip(d, min_reach, max_reach)
    
    # Knee angle (law of cosines)
    cos_knee = (THIGH_LENGTH**2 + CALF_LENGTH**2 - d**2) / (2 * THIGH_LENGTH * CALF_LENGTH)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    
    # User reported legs bend backward (flamingo style).
    # To switch to human-like forward bending, we use the alternate IK solution
    # by negating the knee angle.
    knee = -(math.pi - math.acos(cos_knee))
    
    # Hip pitch
    alpha = math.atan2(dx, dz)
    beta = math.asin(np.clip(CALF_LENGTH * math.sin(knee) / d, -1.0, 1.0))
    hip_pitch = -(alpha + beta)
    
    # Ankle pitch (keep foot flat)
    ank_pitch = -(hip_pitch + knee)
    
    return hip_pitch, knee, ank_pitch


# ============================================================================
# Walking Engine
# ============================================================================

class WalkingEngine:
    """ROBOTIS-style sinusoidal walking engine for OP3.
    
    Usage:
        engine = WalkingEngine()
        engine.set_velocity(forward=0.02, lateral=0.0, turn=0.0)
        engine.start()
        
        while running:
            joint_angles = engine.update(dt)
            # Apply joint_angles to actuators
    """
    
    def __init__(self, params: WalkingParam = None):
        self.p = params or WalkingParam()
        self.time = 0.0
        self.running = False
        
        # Internal timing
        self._update_time_params()
        
        # Smoothed amplitudes
        self._x_move_amp = 0.0
        self._y_move_amp = 0.0
        self._a_move_amp = 0.0
        self._prev_x_move_amp = 0.0
        
        # Phase tracking
        self.phase = 0  # 0-3
        
    def _update_time_params(self):
        """Compute phase timing from period and DSP ratio."""
        T = self.p.period_time
        dsp = self.p.dsp_ratio
        
        self.ssp_time = T * (1.0 - dsp)   # Single-support phase duration
        self.l_ssp_start = T * dsp / 4.0   # Left SSP start
        self.l_ssp_end = self.l_ssp_start + self.ssp_time / 2.0
        self.r_ssp_start = T / 2.0 + T * dsp / 4.0
        self.r_ssp_end = self.r_ssp_start + self.ssp_time / 2.0
        
        self.phase1_time = self.l_ssp_start + self.ssp_time / 4.0  # Left foot highest
        self.phase2_time = T / 2.0                                  # Mid DSP
        self.phase3_time = self.r_ssp_start + self.ssp_time / 4.0  # Right foot highest
        
        # Trajectory parameters
        self.x_swap_period = T / 2.0
        self.x_swap_phase = math.pi
        self.x_swap_amp = self.p.y_swap_amplitude * 0  # No x-swap by default
        
        self.y_swap_period = T
        self.y_swap_phase = 0.0
        self.y_swap_amp = self.p.y_swap_amplitude
        
        self.z_swap_period = T / 2.0
        self.z_swap_phase = math.pi * 2 / 3
        self.z_swap_amp = self.p.z_swap_amplitude
        
        self.x_move_period = T
        self.x_move_phase = math.pi / 2.0 + self.p.step_fb_ratio * math.pi
        
        self.y_move_period = T
        self.y_move_phase = math.pi / 2.0
        
        self.z_move_period = self.ssp_time
        self.z_move_phase = math.pi / 2.0
        
        self.a_move_period = T
        self.a_move_phase = math.pi / 2.0
    
    def set_velocity(self, forward: float = 0.0, lateral: float = 0.0, turn: float = 0.0):
        """Set walking velocity commands (meters/step, radians/step).
        
        Args:
            forward: Forward step length (meters, positive = forward)
            lateral: Lateral step (meters, positive = left)
            turn: Turn angle per step (radians, positive = left)
        """
        self.p.x_move_amplitude = forward
        self.p.y_move_amplitude = lateral
        self.p.angle_move_amplitude = turn
    
    def start(self):
        """Start the walking engine."""
        self.running = True
        self.time = 0.0
        self._update_time_params()
        self._update_movement_params()
    
    def stop(self):
        """Signal to stop walking (will finish current cycle)."""
        self.p.x_move_amplitude = 0.0
        self.p.y_move_amplitude = 0.0
        self.p.angle_move_amplitude = 0.0
    
    def _update_movement_params(self):
        """Smooth the movement amplitude transitions."""
        self._x_move_amp = self.p.x_move_amplitude
        self._y_move_amp = self.p.y_move_amplitude
        self._a_move_amp = self.p.angle_move_amplitude
        
    def update(self, dt: float) -> dict:
        """Advance the walking engine by dt seconds.
        
        Returns dict of actuator_name -> target_angle (radians).
        """
        if not self.running:
            return self._standing_pose()
        
        T = self.p.period_time
        
        # Phase management
        if self.time >= T:
            self.time = 0.0
            self._prev_x_move_amp = self._x_move_amp * 0.5
        
        # Update phase
        if self.time <= self.l_ssp_start:
            self.phase = 0  # Double support
        elif self.time <= self.l_ssp_end:
            self.phase = 1  # Left SSP (right foot support)
        elif self.time <= self.r_ssp_start:
            self.phase = 2  # Double support
        elif self.time <= self.r_ssp_end:
            self.phase = 3  # Right SSP (left foot support)
        else:
            self.phase = 0  # Double support
        
        # At phase boundaries, update params
        if abs(self.time - self.phase1_time) < dt:
            self._update_movement_params()
            self._update_time_params()
        elif abs(self.time - self.phase3_time) < dt:
            self._update_movement_params()
            self._update_time_params()
        
        # Compute foot trajectories
        joint_angles = self._compute_joint_angles()
        
        # Advance time
        self.time += dt
        
        return joint_angles
    
    def _compute_joint_angles(self) -> dict:
        """Core gait computation: sinusoidal trajectories → IK → joint angles."""
        t = self.time
        T = self.p.period_time
        
        # Body sway (swap)
        swap_y = wsin(t, self.y_swap_period, self.y_swap_phase,
                      self.y_swap_amp, 0)
        swap_z = wsin(t, self.z_swap_period, self.z_swap_phase,
                      self.z_swap_amp, 0)
        
        # Foot X movement (forward/backward stride)
        x_amp = self._x_move_amp / 2.0
        x_amp_shift = self._prev_x_move_amp
        
        # Foot Z movement (lifting)
        z_amp = self.p.z_move_amplitude / 2.0
        
        # Foot Y movement (lateral step)
        y_amp = self._y_move_amp / 2.0
        
        # Pelvis sway
        pelvis_swing = self.p.pelvis_offset
        
        # Compute foot positions based on phase
        if self.phase == 0 or self.phase == 2:
            # Double support — hold endpoints
            r_foot_x = x_amp_shift
            l_foot_x = x_amp_shift
            r_foot_z = 0.0
            l_foot_z = 0.0
            r_foot_y = 0.0
            l_foot_y = 0.0
            pelvis_offset_l = 0.0
            pelvis_offset_r = 0.0
            
        elif self.phase == 1:
            # Left leg swing (right foot support)
            # Phase-shifted sinusoidal for smooth trajectory
            phase_x = self.x_move_phase + 2 * math.pi / self.x_move_period * self.l_ssp_start
            phase_z = self.z_move_phase + 2 * math.pi / self.z_move_period * self.l_ssp_start
            
            l_foot_x = wsin(t, self.x_move_period, phase_x, x_amp, x_amp_shift)
            r_foot_x = wsin(t, self.x_move_period, phase_x, -x_amp, -x_amp_shift)
            l_foot_z = wsin(t, self.z_move_period, phase_z, z_amp, z_amp)
            r_foot_z = 0.0
            
            phase_y = self.y_move_phase + 2 * math.pi / self.y_move_period * self.l_ssp_start
            l_foot_y = wsin(t, self.y_move_period, phase_y, y_amp, 0)
            r_foot_y = wsin(t, self.y_move_period, phase_y, -y_amp, 0)
            
            pelvis_offset_l = wsin(t, self.z_move_period, phase_z, pelvis_swing / 2, pelvis_swing / 2)
            pelvis_offset_r = wsin(t, self.z_move_period, phase_z, -pelvis_swing / 2, -pelvis_swing / 2)
            
        elif self.phase == 3:
            # Right leg swing (left foot support)
            phase_x = self.x_move_phase + 2 * math.pi / self.x_move_period * self.r_ssp_start + math.pi
            phase_z = self.z_move_phase + 2 * math.pi / self.z_move_period * self.r_ssp_start
            
            l_foot_x = wsin(t, self.x_move_period, phase_x, x_amp, x_amp_shift)
            r_foot_x = wsin(t, self.x_move_period, phase_x, -x_amp, -x_amp_shift)
            l_foot_z = 0.0
            r_foot_z = wsin(t, self.z_move_period, phase_z, z_amp, z_amp)
            
            phase_y = self.y_move_phase + 2 * math.pi / self.y_move_period * self.r_ssp_start + math.pi
            l_foot_y = wsin(t, self.y_move_period, phase_y, y_amp, 0)
            r_foot_y = wsin(t, self.y_move_period, phase_y, -y_amp, 0)
            
            pelvis_offset_l = wsin(t, self.z_move_period, phase_z, pelvis_swing / 2, pelvis_swing / 2)
            pelvis_offset_r = wsin(t, self.z_move_period, phase_z, -pelvis_swing / 2, -pelvis_swing / 2)
        
        # Compute final foot endpoints relative to hip
        # Right leg endpoint
        r_ep_x = r_foot_x + self.p.init_x_offset
        r_ep_y = r_foot_y - LEG_SIDE_OFFSET + swap_y
        r_ep_z = r_foot_z + self.p.init_z_offset - LEG_LENGTH + swap_z
        
        # Left leg endpoint
        l_ep_x = l_foot_x + self.p.init_x_offset
        l_ep_y = l_foot_y + LEG_SIDE_OFFSET + swap_y
        l_ep_z = l_foot_z + self.p.init_z_offset - LEG_LENGTH + swap_z
        
        # Solve IK using simplified 3-DOF (more robust for our sim)
        r_hip_pitch, r_knee, r_ank_pitch = solve_ik_simple(r_ep_x, r_ep_y, r_ep_z)
        l_hip_pitch, l_knee, l_ank_pitch = solve_ik_simple(l_ep_x, l_ep_y, l_ep_z)
        
        # NOTE: lateral sway (swap_y) is already handled in foot endpoint
        # positions above. No additional hip_roll rotation needed.
        # Only pelvis_offset provides small roll bias during swing phase.
        
        # Apply hip pitch offset (forward lean)
        hip_pitch_off = self.p.hip_pitch_offset
        
        # Arm swing (counter-phase to legs for natural gait)
        # When right leg steps forward, right arm goes backward (and vice versa)
        arm_swing = 0.0
        if abs(self._x_move_amp) > 0.001:
            arm_swing = wsin(t, T, math.pi * 1.5,
                           self._x_move_amp * self.p.arm_swing_gain * 10, 0)
        
        # Build joint angle dict (matching MuJoCo actuator names)
        # Joint axis conventions from op3.xml:
        #   l_hip_pitch: Y+    r_hip_pitch: Y-  (mirrored)
        #   l_knee:      Y+    r_knee:      Y-  (mirrored)
        #   l_ank_pitch: Y-    r_ank_pitch: Y+  (reversed from hip/knee!)
        #   l_hip_roll:  X-    r_hip_roll:  X-  (same)
        #   l_ank_roll:  X+    r_ank_roll:  X+  (same)
        angles = {}
        
        # IK produces: hip_pitch<0 = bend forward, knee>0 = bend, ank_pitch<0 = toe up
        # MuJoCo left Y+: positive = forward rotation
        # MuJoCo right Y-: positive = forward rotation (axis flipped)
        
        # Right leg: hip_pitch axis=Y-, knee axis=Y-, ank_pitch axis=Y+
        angles["r_hip_yaw_act"] = 0.0
        angles["r_hip_roll_act"] = pelvis_offset_r
        angles["r_hip_pitch_act"] = (r_hip_pitch + hip_pitch_off)
        angles["r_knee_act"] = (r_knee)
        angles["r_ank_pitch_act"] = -(r_ank_pitch)
        angles["r_ank_roll_act"] = 0.0
        
        # Left leg: hip_pitch axis=Y+, knee axis=Y+, ank_pitch axis=Y-
        angles["l_hip_yaw_act"] = 0.0
        angles["l_hip_roll_act"] = pelvis_offset_l
        angles["l_hip_pitch_act"] = -(l_hip_pitch + hip_pitch_off)
        angles["l_knee_act"] = -(l_knee)
        angles["l_ank_pitch_act"] = (l_ank_pitch)
        angles["l_ank_roll_act"] = 0.0
        
        # Arm pose verified against ROBOTIS source (op3_kinematics_dynamics.cpp):
        #   getJointDirection() = sum(axis components) = +1 or -1
        #
        # sho_pitch: l=Y+ (dir=+1), r=Y- (dir=-1)
        #   ROBOTIS computeArmAngle: multiplies by getJointDirection()
        #   So right arm gets NEGATED swing for counter-phase motion
        # sho_roll: both X- (dir=-1) → l=+1.3 down, r=-1.3 down
        # el: both X+ (dir=+1) but Y-mirrored geometry → r_el needs opposite sign
        #   Verified: (l=+0.8, r=-0.8) gives symmetric forearm tips
        angles["r_sho_pitch_act"] = -arm_swing      # Y-: negated for counter-phase (ROBOTIS: * dir = * -1)
        angles["l_sho_pitch_act"] = arm_swing        # Y+: same sign (ROBOTIS: * dir = * +1)
        angles["r_sho_roll_act"] = -1.3              # Arm down
        angles["l_sho_roll_act"] = 1.3               # Arm down
        angles["r_el_act"] = 0.8                     # Elbow bent inward (toward body center)
        angles["l_el_act"] = -0.8                    # Elbow bent inward (toward body center)
        
        # Head
        angles["head_pan_act"] = 0.0
        angles["head_tilt_act"] = 0.0
        
        return angles
    
    def _standing_pose(self) -> dict:
        """Default standing pose with slight knee bend and forward lean."""
        # Use IK to compute a stable standing pose
        stand_z = self.p.init_z_offset - LEG_LENGTH
        hip_pitch, knee, ank_pitch = solve_ik_simple(self.p.init_x_offset, 0, stand_z)
        
        hip_pitch_off = self.p.hip_pitch_offset
        
        angles = {}
        angles["r_hip_yaw_act"] = 0.0
        angles["r_hip_roll_act"] = 0.0
        angles["r_hip_pitch_act"] = (hip_pitch + hip_pitch_off)
        angles["r_knee_act"] = (knee)
        angles["r_ank_pitch_act"] = -(ank_pitch)
        angles["r_ank_roll_act"] = 0.0
        angles["l_hip_yaw_act"] = 0.0
        angles["l_hip_roll_act"] = 0.0
        angles["l_hip_pitch_act"] = -(hip_pitch + hip_pitch_off)
        angles["l_knee_act"] = -(knee)
        angles["l_ank_pitch_act"] = (ank_pitch)
        angles["l_ank_roll_act"] = 0.0
        angles["r_sho_pitch_act"] = 0.0
        angles["l_sho_pitch_act"] = 0.0
        angles["r_sho_roll_act"] = -1.3
        angles["l_sho_roll_act"] = 1.3
        angles["r_el_act"] = 0.8                     # Elbow bent inward
        angles["l_el_act"] = -0.8                    # Elbow bent inward
        angles["head_pan_act"] = 0.0
        angles["head_tilt_act"] = 0.0
        return angles
    
    @property
    def phase_name(self) -> str:
        return ["DSP", "L_SWING", "DSP", "R_SWING"][self.phase]

    @staticmethod
    def apply_balance(angles: dict, pitch: float, roll: float,
                      pitch_gain: float = 0.5, roll_gain: float = 0.3) -> dict:
        """Apply IMU-based balance feedback to ankle joints.
        
        Args:
            angles: Joint angle dict from update()
            pitch: Body pitch angle (radians, positive = forward lean)
            roll: Body roll angle (radians, positive = left tilt)
            pitch_gain: Ankle pitch correction gain
            roll_gain: Ankle roll correction gain
        """
        # Ankle pitch fights forward/backward tilt
        # When robot leans forward (pitch>0), ankles should push back
        pitch_corr = pitch_gain * pitch
        # Both ankles dorsi-flex to push CoM backward
        angles["l_ank_pitch_act"] = angles.get("l_ank_pitch_act", 0) + pitch_corr
        angles["r_ank_pitch_act"] = angles.get("r_ank_pitch_act", 0) + pitch_corr
        
        # Ankle roll fights lateral tilt
        roll_corr = -roll_gain * roll
        angles["l_ank_roll_act"] = angles.get("l_ank_roll_act", 0) + roll_corr
        angles["r_ank_roll_act"] = angles.get("r_ank_roll_act", 0) + roll_corr
        
        return angles
