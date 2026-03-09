"""
=============================================================================
  OpenServoSim - Milestone 2: Breathing Motion (Fixed)
=============================================================================

  OP3 does gentle oscillations while standing.
  
  Key insight: OP3 has very weak actuators (real Dynamixel XM430 specs),
  so large squat motions cause the robot to fall.  We keep the motion
  VERY gentle — small hip roll oscillation ("swaying") and tiny knee
  bends, just like a real servo robot's idle animation.

  Run:
      python examples/02_breathing.py
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
        pr, "models", "reference", "robotis_op3", "scene.xml"
    )


def body_height(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    return data.xpos[bid][2] if bid >= 0 else 0


def breathing(data, t):
    """
    Gentle breathing animation.
    
    Since OP3 actuators are weak (matching real servo specs), we:
    1. Stay very close to zero-pose (the only stable equilibrium)
    2. Oscillate hip roll slowly (side-to-side weight shift)
    3. Add tiny arm motion for visual life
    
    Amplitude is kept < 0.1 rad to avoid toppling.
    """
    freq = 0.5  # Hz
    phase = 2 * np.pi * freq * t

    # Gentle lateral sway via hip roll (both legs tilt together)
    sway = 0.06 * np.sin(phase)
    ctrl(data, "l_hip_roll_act", sway)
    ctrl(data, "r_hip_roll_act", sway)
    ctrl(data, "l_ank_roll_act", -sway)
    ctrl(data, "r_ank_roll_act", -sway)

    # Very gentle arm swing 
    arm = 0.15 * np.sin(phase * 0.5)
    ctrl(data, "l_sho_pitch_act",  0.2 + arm)
    ctrl(data, "r_sho_pitch_act", -(0.2 + arm))

    # Elbow slight bend
    ctrl(data, "l_el_act", -0.3)
    ctrl(data, "r_el_act",  0.3)

    # Head gentle nod
    ctrl(data, "head_tilt_act", 0.05 * np.sin(phase * 0.3))

    # All other joints hold at zero
    for name in ["head_pan_act",
                 "l_sho_roll_act", "r_sho_roll_act",
                 "l_hip_yaw_act", "r_hip_yaw_act",
                 "l_hip_pitch_act", "r_hip_pitch_act",
                 "l_knee_act", "r_knee_act",
                 "l_ank_pitch_act", "r_ank_pitch_act"]:
        ctrl(data, name, 0.0)

    return sway


def main():
    print("=" * 60)
    print("  OpenServoSim - Milestone 2: Breathing (Idle Sway)")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    build_act_map(model)

    # Settle into zero stance
    print("  Settling into standing pose...")
    for i in range(model.nu):
        data.ctrl[i] = 0.0
    for _ in range(2000):
        mujoco.mj_step(model, data)
    print(f"  Height: {body_height(model, data)*1000:.0f}mm")

    # Control loop
    ctrl_dt = 1.0 / 50.0
    steps_per = int(ctrl_dt / model.opt.timestep)

    print("  Starting gentle sway animation (0.5 Hz)...")
    print("  Close the MuJoCo window or press ESC to exit.")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = -2.0
        while viewer.is_running():
            t0 = time.time()
            sway = breathing(data, data.time)

            for _ in range(steps_per):
                mujoco.mj_step(model, data)

            if data.time - last_print >= 2.0:
                last_print = data.time
                h = body_height(model, data)
                status = "Standing" if h > 0.2 else "FALLEN"
                print(f"  t={data.time:5.1f}s  height={h*1000:.0f}mm  sway={np.degrees(sway):+.1f}deg  [{status}]")

            viewer.sync()
            sl = ctrl_dt - (time.time() - t0)
            if sl > 0:
                time.sleep(sl)

    print("  Done!")
    print("  Next: python examples/03_simple_walk.py")


if __name__ == "__main__":
    main()
