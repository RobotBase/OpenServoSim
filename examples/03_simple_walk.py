"""
=============================================================================
  OpenServoSim - Milestone 3: Simple Walk (with Balance Feedback)
=============================================================================

  OP3 attempts to walk using a simple approach:
  1. Shift weight onto one foot (hip roll)
  2. Swing the free leg forward (hip pitch)
  3. Put foot down, shift weight to other side
  4. Repeat

  Because OP3's actuators are weak (matching real servo specs),
  we use a conservative approach with very small step sizes
  and IMU-based pitch/roll feedback for active balance.

  Run:
      python examples/03_simple_walk.py
=============================================================================
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer


ACT = {}

def build_act_map(model):
    global ACT
    ACT = {model.actuator(i).name: i for i in range(model.nu)}

def ctrl(data, name, val):
    if name in ACT:
        data.ctrl[ACT[name]] = val

def get_model_path():
    sd = os.path.dirname(os.path.abspath(__file__))
    pr = os.path.dirname(sd)
    p = os.path.join(pr, "models", "reference", "robotis_op3", "scene_enhanced.xml")
    return p if os.path.exists(p) else os.path.join(
        pr, "models", "reference", "robotis_op3", "scene.xml")

def get_imu(model, data):
    """Extract pitch and roll from the body quaternion."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    qw, qx, qy, qz = data.xquat[bid]
    # Pitch (forward/backward tilt)
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    # Roll (lateral tilt)
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    return pitch, roll

def body_height(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    return data.xpos[bid][2]

def body_x(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    return data.xpos[bid][0]


# ===========================================================================
# Walk State Machine
# ===========================================================================
STATE_STAND    = 0
STATE_SHIFT_R  = 1  # Shift weight to right foot
STATE_SWING_L  = 2  # Swing left leg forward
STATE_LAND_L   = 3  # Land left foot
STATE_SHIFT_L  = 4  # Shift weight to left foot
STATE_SWING_R  = 5  # Swing right leg forward
STATE_LAND_R   = 6  # Land right foot

STATE_NAMES = ["STAND", "SHIFT_R", "SWING_L", "LAND_L",
               "SHIFT_L", "SWING_R", "LAND_R"]

# Timing (in seconds)
STAND_DURATION = 1.5
SHIFT_DURATION = 0.8
SWING_DURATION = 0.6
LAND_DURATION  = 0.4

# Gait amplitudes (kept very small for stability)
SHIFT_ROLL = 0.08       # Hip roll for weight shift (rad)
SWING_PITCH = 0.06      # Hip pitch for forward swing (rad)
LIFT_KNEE = 0.08        # Extra knee bend for foot clearance (rad)
ARM_SWING_AMP = 0.15    # Arm swing amplitude (rad)

# Balance feedback gains
PITCH_GAIN = 0.3  # How much pitch error drives ankle pitch
ROLL_GAIN = 0.3   # How much roll error drives ankle roll


class WalkController:
    def __init__(self):
        self.state = STATE_STAND
        self.state_timer = 0.0
        self.step_count = 0
        self.max_steps = 12  # Walk this many steps then stop

    def smooth_interp(self, t, duration, start, end):
        """Smooth interpolation using cosine ease."""
        s = np.clip(t / duration, 0, 1)
        s = 0.5 * (1 - np.cos(np.pi * s))  # Cosine ease-in-ease-out
        return start + (end - start) * s

    def update(self, data, model, dt):
        self.state_timer += dt
        pitch, roll = get_imu(model, data)

        # Default: hold zero
        targets = {name: 0.0 for name in ACT}

        # Arms at sides
        targets["l_sho_roll_act"] = -0.3
        targets["r_sho_roll_act"] =  0.3
        targets["l_el_act"] = -0.4
        targets["r_el_act"] =  0.4

        t = self.state_timer

        if self.state == STATE_STAND:
            # Just stand, prepare to walk
            if t >= STAND_DURATION:
                self._next_state(STATE_SHIFT_R)

        elif self.state == STATE_SHIFT_R:
            # Shift weight onto right foot
            s = self.smooth_interp(t, SHIFT_DURATION, 0, SHIFT_ROLL)
            targets["l_hip_roll_act"] = s
            targets["r_hip_roll_act"] = s
            targets["l_ank_roll_act"] = -s
            targets["r_ank_roll_act"] = -s
            if t >= SHIFT_DURATION:
                self._next_state(STATE_SWING_L)

        elif self.state == STATE_SWING_L:
            # Hold weight on right, swing left leg forward
            targets["l_hip_roll_act"] = SHIFT_ROLL
            targets["r_hip_roll_act"] = SHIFT_ROLL
            targets["l_ank_roll_act"] = -SHIFT_ROLL
            targets["r_ank_roll_act"] = -SHIFT_ROLL

            # Left leg swing: hip forward, knee lift
            swing = self.smooth_interp(t, SWING_DURATION, 0, 1)
            targets["l_hip_pitch_act"] = SWING_PITCH * swing
            targets["l_knee_act"] = LIFT_KNEE * np.sin(np.pi * swing)
            # Counter-swing right hip slightly back
            targets["r_hip_pitch_act"] = SWING_PITCH * 0.3 * swing

            # Arm counter-swing
            targets["l_sho_pitch_act"] = -ARM_SWING_AMP * swing
            targets["r_sho_pitch_act"] =  ARM_SWING_AMP * swing

            if t >= SWING_DURATION:
                self._next_state(STATE_LAND_L)

        elif self.state == STATE_LAND_L:
            # Keep weight shifted, lower left foot
            targets["l_hip_roll_act"] = SHIFT_ROLL
            targets["r_hip_roll_act"] = SHIFT_ROLL
            targets["l_ank_roll_act"] = -SHIFT_ROLL
            targets["r_ank_roll_act"] = -SHIFT_ROLL
            targets["l_hip_pitch_act"] = SWING_PITCH
            targets["r_hip_pitch_act"] = SWING_PITCH * 0.3

            if t >= LAND_DURATION:
                self.step_count += 1
                if self.step_count >= self.max_steps:
                    self._next_state(STATE_STAND)
                else:
                    self._next_state(STATE_SHIFT_L)

        elif self.state == STATE_SHIFT_L:
            # Shift weight left (negative roll)
            s = self.smooth_interp(t, SHIFT_DURATION, SHIFT_ROLL, -SHIFT_ROLL)
            targets["l_hip_roll_act"] = s
            targets["r_hip_roll_act"] = s
            targets["l_ank_roll_act"] = -s
            targets["r_ank_roll_act"] = -s
            # Keep hip pitch from previous step
            targets["l_hip_pitch_act"] = SWING_PITCH
            targets["r_hip_pitch_act"] = SWING_PITCH * 0.3

            if t >= SHIFT_DURATION:
                self._next_state(STATE_SWING_R)

        elif self.state == STATE_SWING_R:
            # Hold weight on left, swing right leg forward
            targets["l_hip_roll_act"] = -SHIFT_ROLL
            targets["r_hip_roll_act"] = -SHIFT_ROLL
            targets["l_ank_roll_act"] = SHIFT_ROLL
            targets["r_ank_roll_act"] = SHIFT_ROLL

            swing = self.smooth_interp(t, SWING_DURATION, 0, 1)
            targets["r_hip_pitch_act"] = -SWING_PITCH * swing
            targets["r_knee_act"] = -LIFT_KNEE * np.sin(np.pi * swing)
            targets["l_hip_pitch_act"] = SWING_PITCH + (-SWING_PITCH * 0.3 * swing)

            targets["l_sho_pitch_act"] =  ARM_SWING_AMP * swing
            targets["r_sho_pitch_act"] = -ARM_SWING_AMP * swing

            if t >= SWING_DURATION:
                self._next_state(STATE_LAND_R)

        elif self.state == STATE_LAND_R:
            targets["l_hip_roll_act"] = -SHIFT_ROLL
            targets["r_hip_roll_act"] = -SHIFT_ROLL
            targets["l_ank_roll_act"] = SHIFT_ROLL
            targets["r_ank_roll_act"] = SHIFT_ROLL
            targets["r_hip_pitch_act"] = -SWING_PITCH

            if t >= LAND_DURATION:
                self.step_count += 1
                if self.step_count >= self.max_steps:
                    self._next_state(STATE_STAND)
                else:
                    self._next_state(STATE_SHIFT_R)

        # --- Balance feedback (active stabilization) ---
        # Ankle pitch fights forward/backward tilt
        pitch_correction = -PITCH_GAIN * pitch
        targets["l_ank_pitch_act"] = targets.get("l_ank_pitch_act", 0) + pitch_correction
        targets["r_ank_pitch_act"] = targets.get("r_ank_pitch_act", 0) - pitch_correction

        # Ankle roll fights lateral tilt
        roll_correction = -ROLL_GAIN * roll
        targets["l_ank_roll_act"] = targets.get("l_ank_roll_act", 0) + roll_correction
        targets["r_ank_roll_act"] = targets.get("r_ank_roll_act", 0) + roll_correction

        # Apply all targets
        for name, val in targets.items():
            ctrl(data, name, val)

        return self.state

    def _next_state(self, new_state):
        self.state = new_state
        self.state_timer = 0.0


def main():
    print("=" * 60)
    print("  OpenServoSim - Milestone 3: Simple Walk")
    print("  (State Machine + Balance Feedback)")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    build_act_map(model)

    # Settle
    print("  Settling...")
    for i in range(model.nu):
        data.ctrl[i] = 0.0
    for _ in range(2000):
        mujoco.mj_step(model, data)
    print(f"  Height: {body_height(model, data)*1000:.0f}mm")

    walker = WalkController()
    ctrl_dt = 1.0 / 50.0
    steps_per = int(ctrl_dt / model.opt.timestep)

    print(f"\n  Walking for {walker.max_steps} steps...")
    print("  " + "-" * 60)
    print(f"  {'Time':>5s}  {'State':<10s}  {'Height':>7s}  {'X-pos':>8s}  {'Pitch':>7s}  {'Steps':>5s}")
    print("  " + "-" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = -1.5
        while viewer.is_running():
            t0 = time.time()

            state = walker.update(data, model, ctrl_dt)

            for _ in range(steps_per):
                mujoco.mj_step(model, data)

            if data.time - last_print >= 1.5:
                last_print = data.time
                h = body_height(model, data)
                x = body_x(model, data)
                pitch, roll = get_imu(model, data)
                sname = STATE_NAMES[state]
                print(
                    f"  {data.time:5.1f}s  {sname:<10s}  "
                    f"{h*1000:6.0f}mm  {x*1000:+7.1f}mm  "
                    f"{np.degrees(pitch):+6.1f}d  {walker.step_count:5d}"
                )

            viewer.sync()
            sl = ctrl_dt - (time.time() - t0)
            if sl > 0:
                time.sleep(sl)

    print(f"\n  Done! Total distance: {body_x(model, data)*1000:.1f}mm in {walker.step_count} steps")


if __name__ == "__main__":
    main()
