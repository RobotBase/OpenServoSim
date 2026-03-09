"""
=============================================================================
  OpenServoSim - UVC Walking Demo
=============================================================================

  Unified walking + UVC balance.

  Speed presets: normal (default), fast, sprint
  Controls (viewer mode): P = push, close window to exit

  Run:
      python examples/05_uvc_walk.py                        # normal speed, viewer
      python examples/05_uvc_walk.py --speed fast            # fast, viewer
      python examples/05_uvc_walk.py --speed sprint          # sprint, viewer
      python examples/05_uvc_walk.py --headless              # headless diagnostics
      python examples/05_uvc_walk.py --headless --push-at 5  # with push
=============================================================================
"""

import os
import sys
import time
import argparse
import ctypes
import numpy as np
import math
import mujoco
import mujoco.viewer

# Force NVIDIA GPU (defeats Meta Virtual Monitor)
try:
    ctypes.CDLL("nvapi64.dll")
except OSError:
    pass
if sys.platform == "win32":
    try:
        ctypes.windll.kernel32.SetEnvironmentVariableW("SHIM_MCCOMPAT", "0x800000001")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.uvc_walking import UVCWalkingEngine, UVCParam
from controllers.robotis_walking import WalkingParam, WalkingEngine
import math


# ===== Speed Presets =====

SPEED_PRESETS = {
    "normal": {
        "walk": WalkingParam(
            x_move_amplitude=0.025,
            period_time=0.600,
            z_move_amplitude=0.025,
            dsp_ratio=0.20,
            y_swap_amplitude=0.020,
            z_swap_amplitude=0.004,
            arm_swing_gain=0.5,
        ),
        "uvc": UVCParam(
            gain_roll=0.20, gain_pitch=0.15,
            dead_zone=0.05, warmup_time=2.0,
        ),
        "forward": 0.025,
    },
    "fast": {
        "walk": WalkingParam(
            x_move_amplitude=0.040,    # 60% bigger step
            period_time=0.550,
            z_move_amplitude=0.030,
            dsp_ratio=0.18,
            y_swap_amplitude=0.024,
            z_swap_amplitude=0.005,
            arm_swing_gain=0.6,
            pelvis_offset=math.radians(0.8),
            balance_ankle_pitch_gain=1.1,
            balance_ankle_roll_gain=0.85,
        ),
        "uvc": UVCParam(
            gain_roll=0.25, gain_pitch=0.18,
            dead_zone=0.045,
            roll_scale=0.18, pitch_scale=0.12,
            max_correction=0.10,
            warmup_time=1.5,
        ),
        "forward": 0.040,
    },
    "sprint": {
        "walk": WalkingParam(
            x_move_amplitude=0.041,    # Absolute max limits 
            period_time=0.560,         # Slower to allow wider swing
            z_move_amplitude=0.032,
            dsp_ratio=0.18,
            y_swap_amplitude=0.024,
            z_swap_amplitude=0.005,
            arm_swing_gain=0.8,
            pelvis_offset=math.radians(1.0),
            balance_ankle_pitch_gain=1.2,
            balance_ankle_roll_gain=0.9,
            balance_hip_roll_gain=0.4,
        ),
        "uvc": UVCParam(
            gain_roll=0.28, gain_pitch=0.18,
            dead_zone=0.04,
            roll_scale=0.18, pitch_scale=0.13,
            max_correction=0.12,
            warmup_time=1.0,
        ),
        "forward": 0.041,
    },
    "stride": {
        "walk": WalkingParam(
            x_move_amplitude=0.042,    # Widest step before instability
            period_time=0.580,         # Slower cadence for balance
            z_move_amplitude=0.035,    
            dsp_ratio=0.20,
            y_swap_amplitude=0.028,    
            z_swap_amplitude=0.006,    
            arm_swing_gain=0.9,        
            pelvis_offset=math.radians(1.2),
            balance_ankle_pitch_gain=1.2,
            balance_ankle_roll_gain=1.0,
            balance_hip_roll_gain=0.45,
            balance_knee_gain=0.35,
        ),
        "uvc": UVCParam(
            gain_roll=0.28, gain_pitch=0.18,
            dead_zone=0.04,
            roll_scale=0.20, pitch_scale=0.14,
            max_correction=0.12,
            roll_decay=0.91, pitch_decay=0.89,
            warmup_time=1.5,
        ),
        "forward": 0.042,
    },
    "ninja": {
        # User requested: bent knees, active arms, fast walking
        # Strategy: "Scurrying" - very low stance, very fast cadence, active arms
        "walk": WalkingParam(
            init_z_offset=0.040,       # Deep crouch (40mm knee bend)
            x_move_amplitude=0.035,    # Moderate step to maintain stability
            period_time=0.460,         # VERY fast cadence (scurrying)
            z_move_amplitude=0.030,    # Fast foot pickup
            dsp_ratio=0.15,
            y_swap_amplitude=0.024,
            z_swap_amplitude=0.005,
            arm_swing_gain=1.2,        # Active arms
            pelvis_offset=math.radians(1.5), # More pelvis tilt for dynamic look
            balance_ankle_pitch_gain=1.2,
            balance_ankle_roll_gain=0.9,
            balance_hip_roll_gain=0.4,
            balance_knee_gain=0.5,     # Extra knee damping for crouch
        ),
        "uvc": UVCParam(
            gain_roll=0.30, gain_pitch=0.20,
            dead_zone=0.04,
            roll_scale=0.20, pitch_scale=0.14,
            max_correction=0.14,
            roll_decay=0.91, pitch_decay=0.89,
            warmup_time=1.0,
        ),
        "forward": 0.035,
    },
}


def get_model_path():
    pr = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(pr, "models", "reference", "robotis_op3", "scene_enhanced.xml")
    return p if os.path.exists(p) else os.path.join(
        pr, "models", "reference", "robotis_op3", "scene.xml")


def body_height(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    return data.xpos[bid][2]


def body_pos(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    return data.xpos[bid].copy()


def get_imu(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    qw, qx, qy, qz = data.xquat[bid]
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    return pitch, roll


def build_act_map(model):
    return {model.actuator(i).name: i for i in range(model.nu)}


def apply_push(model, data, force_x=5.0, force_y=3.0):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    data.xfrc_applied[bid] = [force_x, force_y, 0, 0, 0, 0]


def clear_push(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    data.xfrc_applied[bid] = [0, 0, 0, 0, 0, 0]


# ===== Run Modes =====

def run_headless(preset_name, duration=20.0, push_at=None):
    preset = SPEED_PRESETS[preset_name]
    walk_p = preset["walk"]
    uvc_p = preset["uvc"]
    fwd_vel = preset["forward"]

    print("=" * 70)
    print(f"  UVC Walking — [{preset_name.upper()}] mode — Headless Diagnostics")
    print("=" * 70)
    print(f"  Period: {walk_p.period_time*1000:.0f}ms  Step: {fwd_vel*1000:.0f}mm  "
          f"Foot lift: {walk_p.z_move_amplitude*1000:.0f}mm  DSP: {walk_p.dsp_ratio:.0%}")

    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    act_map = build_act_map(model)

    engine = UVCWalkingEngine(walk_p, uvc_p)

    # Settle
    standing = engine.walk._standing_pose()
    for name, val in standing.items():
        if name in act_map:
            data.ctrl[act_map[name]] = val
    for _ in range(3000):
        mujoco.mj_step(model, data)

    h0 = body_height(model, data)
    print(f"  Standing height: {h0*1000:.0f}mm")

    engine.set_velocity(forward=fwd_vel)
    engine.start()

    ctrl_dt = 1.0 / 50.0
    steps_per = int(ctrl_dt / model.opt.timestep)
    settle_time = 6.0

    # Speed measurement: record position at t=settle+2s and at end
    speed_start_t = settle_time + 2.0
    speed_start_x = None

    if push_at:
        print(f"  Push at t={push_at:.1f}s")

    min_h = 999.0
    max_h = 0.0
    push_applied = False
    push_cleared = False
    fell = False
    samples = []
    prev_pos = None
    path_length = 0.0      # Total XY distance traveled
    path_started = False

    while data.time < settle_time + duration:
        pitch, roll = get_imu(model, data)

        sim_t = data.time - settle_time
        if push_at and not push_applied and sim_t >= push_at:
            print(f"  >>> PUSH at t={sim_t:.1f}s")
            apply_push(model, data)
            push_applied = True
        if push_applied and not push_cleared and sim_t >= push_at + 0.1:
            clear_push(model, data)
            push_cleared = True

        angles = engine.update(ctrl_dt, pitch, roll)
        WalkingEngine.apply_balance(angles, pitch, roll)
        for name, val in angles.items():
            if name in act_map:
                data.ctrl[act_map[name]] = val
        for _ in range(steps_per):
            mujoco.mj_step(model, data)

        h = body_height(model, data)
        pos = body_pos(model, data)
        min_h = min(min_h, h)
        max_h = max(max_h, h)

        # Path length measurement (after 2s warmup)
        if data.time >= speed_start_t:
            if not path_started:
                path_started = True
                speed_start_x = pos[0]
            if prev_pos is not None and h > 0.18:
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                path_length += math.sqrt(dx*dx + dy*dy)
            prev_pos = pos.copy()

        if h < 0.18 and not fell:
            fell = True
            print(f"  FELL at t={data.time:.1f}s (h={h*1000:.0f}mm)")

        pitch_a, roll_a = get_imu(model, data)
        samples.append({
            "t": data.time, "h": h, "pitch": pitch_a, "roll": roll_a,
            "x": pos[0], "y": pos[1], "phase": engine.phase_name,
        })

    # === Speed calculation (path length) ===
    end_x = body_pos(model, data)[0]
    meas_time = data.time - speed_start_t if path_started else 0
    net_disp = (end_x - speed_start_x) if speed_start_x is not None else 0
    path_speed = path_length / meas_time if meas_time > 0 else 0

    print(f"\n  === RESULTS ===")
    print(f"  Path speed: {path_speed*1000:.1f} mm/s  (path={path_length*1000:.0f}mm in {meas_time:.1f}s)")
    print(f"  Net X:      {net_disp*1000:+.0f}mm  (robot walks in arc)")
    print(f"  Height:     min={min_h*1000:.0f}mm  max={max_h*1000:.0f}mm")
    print(f"  Pitch:      [{np.degrees(min(s['pitch'] for s in samples)):+.1f}, "
          f"{np.degrees(max(s['pitch'] for s in samples)):+.1f}] deg")
    print(f"  Roll:       [{np.degrees(min(s['roll'] for s in samples)):+.1f}, "
          f"{np.degrees(max(s['roll'] for s in samples)):+.1f}] deg")

    # Timeline
    print(f"\n  Timeline:")
    print(f"  {'t':>5s}  {'Phase':<8s}  {'H':>5s}  {'Pitch':>6s}  {'Roll':>6s}  {'X':>7s}  {'Y':>7s}")
    last_t = -1.0
    for s in samples:
        if s["t"] - last_t >= 2.0:
            last_t = s["t"]
            print(f"  {s['t']:5.1f}s {s['phase']:<8s} "
                  f"{s['h']*1000:4.0f}mm {np.degrees(s['pitch']):+5.1f}d "
                  f"{np.degrees(s['roll']):+5.1f}d {s['x']*1000:+6.0f}mm {s['y']*1000:+6.0f}mm")

    passed = not fell and min_h >= 0.200
    print(f"\n  Result: {'[PASS]' if passed else '[FAIL]'}  "
          f"Speed: {path_speed*1000:.1f} mm/s")
    return passed


def run_viewer(preset_name):
    preset = SPEED_PRESETS[preset_name]
    walk_p = preset["walk"]
    uvc_p = preset["uvc"]
    fwd_vel = preset["forward"]

    print("=" * 70)
    print(f"  OpenServoSim — UVC Walking [{preset_name.upper()}]  (P=push)")
    print("=" * 70)
    print(f"  Period: {walk_p.period_time*1000:.0f}ms  Step: {fwd_vel*1000:.0f}mm  "
          f"Foot lift: {walk_p.z_move_amplitude*1000:.0f}mm")

    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    act_map = build_act_map(model)

    engine = UVCWalkingEngine(walk_p, uvc_p)
    walk_started = False

    standing = engine.walk._standing_pose()
    for name, val in standing.items():
        if name in act_map:
            data.ctrl[act_map[name]] = val
    for _ in range(3000):
        mujoco.mj_step(model, data)
    print(f"  Standing height: {body_height(model, data)*1000:.0f}mm")
    print(f"  Walking starts in 2s...")

    ctrl_dt = 1.0 / 50.0
    steps_per = int(ctrl_dt / model.opt.timestep)

    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    except Exception as e:
        print(f"\n  [ERROR] Viewer failed: {e}")
        print("  Try: python examples/05_uvc_walk.py --headless")
        sys.exit(1)

    with viewer_ctx as viewer:
        last_print = -2.0
        start_x = None
        walk_start_t = None

        while viewer.is_running():
            t0 = time.time()
            pitch, roll = get_imu(model, data)

            if not walk_started and data.time > 2.0:
                walk_started = True
                walk_start_t = data.time
                engine.set_velocity(forward=fwd_vel)
                engine.start()
                print(f"  Walking started! [{preset_name.upper()}]")

            if walk_started:
                angles = engine.update(ctrl_dt, pitch, roll)
                WalkingEngine.apply_balance(angles, pitch, roll)
            else:
                angles = engine.walk._standing_pose()

            for name, val in angles.items():
                if name in act_map:
                    data.ctrl[act_map[name]] = val
            for _ in range(steps_per):
                mujoco.mj_step(model, data)

            h = body_height(model, data)
            pos = body_pos(model, data)

            if walk_started and start_x is None and data.time > walk_start_t + 2.0:
                start_x = pos[0]

            if data.time - last_print >= 2.0:
                last_print = data.time
                # Speed measurement
                if start_x is not None:
                    elapsed = data.time - walk_start_t - 2.0
                    spd = (pos[0] - start_x) / elapsed * 1000 if elapsed > 0 else 0
                    spd_str = f"speed={spd:.0f}mm/s"
                else:
                    spd_str = "measuring..."

                uvc = engine.get_uvc_info()
                pitch_a, roll_a = get_imu(model, data)
                print(
                    f"  t={data.time:5.1f}s  {engine.phase_name:<8s}  "
                    f"h={h*1000:.0f}mm  x={pos[0]*1000:+.0f}mm  "
                    f"p={np.degrees(pitch_a):+.1f}  r={np.degrees(roll_a):+.1f}  "
                    f"[{spd_str}]"
                )
                if h < 0.10:
                    break

            viewer.sync()
            sleep_time = ctrl_dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="UVC Walking Demo")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", choices=["normal", "fast", "sprint", "stride", "ninja"], default="normal",
                       help="Speed preset: normal, fast, sprint, stride, ninja")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--push-at", type=float, default=None)
    args = parser.parse_args()

    if args.headless:
        run_headless(args.speed, args.duration, args.push_at)
    else:
        run_viewer(args.speed)


if __name__ == "__main__":
    main()
