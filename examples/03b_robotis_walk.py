"""
=============================================================================
  OpenServoSim - Milestone 3b: ROBOTIS-Style Walking
=============================================================================

  Uses the ported ROBOTIS walking engine to generate stable walking.
  The engine computes foot trajectories in Cartesian space using sinusoidal
  functions, then solves IK to get joint angles.

  Controls (viewer mode):
    Auto-starts walking after 2s settling.
    Close window to exit.

  Run:
      python examples/03b_robotis_walk.py               # viewer
      python examples/03b_robotis_walk.py --headless     # headless diagnostics
=============================================================================
"""

import os
import sys
import time
import argparse
import ctypes

# ── Force NVIDIA discrete GPU (defeats Meta Virtual Monitor / Optimus) ──
# These exported symbols tell the NVIDIA driver to prefer the discrete GPU.
# They must exist before any OpenGL context is created.
try:
    ctypes.CDLL("nvapi64.dll")   # preload NVIDIA API
except OSError:
    pass
# Optimus global hint (DWORD export)
if sys.platform == "win32":
    try:
        ctypes.windll.kernel32.SetEnvironmentVariableW(
            "SHIM_MCCOMPAT", "0x800000001"
        )
    except Exception:
        pass

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.robotis_walking import WalkingEngine, WalkingParam


def get_model_path():
    sd = os.path.dirname(os.path.abspath(__file__))
    pr = os.path.dirname(sd)
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


# ===== Diagnostics =====

class WalkDiagnostics:
    """Tracks walking metrics and auto-detects falls with root-cause analysis."""

    FALL_THRESHOLD = 0.18   # Below this height = fallen (meters)
    PITCH_DANGER = 0.35     # ~20 deg pitch = danger zone (radians)

    def __init__(self):
        self.samples = []      # list of dicts per control step
        self.fall_time = None
        self.fall_cause = None

    def record(self, sim_time, height, pitch, roll, phase, angles, x_pos):
        sample = {
            "t": sim_time,
            "h": height,
            "pitch": pitch,
            "roll": roll,
            "phase": phase,
            "x": x_pos,
        }
        # Track key joint angles
        for key in ["l_hip_pitch_act", "r_hip_pitch_act",
                     "l_knee_act", "r_knee_act",
                     "l_ank_pitch_act", "r_ank_pitch_act",
                     "l_hip_roll_act", "r_hip_roll_act",
                     "l_ank_roll_act", "r_ank_roll_act"]:
            sample[key] = angles.get(key, 0.0)
        self.samples.append(sample)

        # Detect fall
        if self.fall_time is None and height < self.FALL_THRESHOLD:
            self.fall_time = sim_time
            self._analyze_fall()

    def _analyze_fall(self):
        """Analyze what caused the fall by looking at history."""
        if len(self.samples) < 5:
            self.fall_cause = "Fell immediately at startup"
            return

        # Look at the last 2 seconds before fall
        fall_idx = len(self.samples) - 1
        lookback = min(100, fall_idx)  # ~2s at 50Hz
        window = self.samples[fall_idx - lookback:fall_idx + 1]

        # Check pitch trend
        pitches = [s["pitch"] for s in window]
        max_pitch = max(pitches)
        min_pitch = min(pitches)
        final_pitch = pitches[-1]

        # Check roll trend
        rolls = [s["roll"] for s in window]
        max_roll_abs = max(abs(r) for r in rolls)

        # Check height trend
        heights = [s["h"] for s in window]
        h_start = heights[0]
        h_end = heights[-1]
        h_drop = h_start - h_end

        # Determine cause
        if abs(final_pitch) > self.PITCH_DANGER:
            direction = "FORWARD (face-plant)" if final_pitch < 0 else "BACKWARD"
            # Find when pitch first exceeded danger
            danger_time = None
            for s in window:
                if abs(s["pitch"]) > self.PITCH_DANGER * 0.5:
                    danger_time = s["t"]
                    break
            self.fall_cause = (
                f"Pitch topple {direction}: "
                f"final_pitch={np.degrees(final_pitch):+.1f}deg, "
                f"danger started at t={danger_time:.1f}s" if danger_time else ""
            )
        elif max_roll_abs > self.PITCH_DANGER:
            side = "LEFT" if rolls[-1] > 0 else "RIGHT"
            self.fall_cause = f"Roll topple to {side}: max_roll={np.degrees(max_roll_abs):.1f}deg"
        elif h_drop > 0.05:
            self.fall_cause = f"Height collapse: dropped {h_drop*1000:.0f}mm (actuator weakness)"
        else:
            self.fall_cause = f"Unknown: h={h_end*1000:.0f}mm, pitch={np.degrees(final_pitch):+.1f}deg"

    def print_report(self, final_time):
        """Print comprehensive diagnostics report."""
        if not self.samples:
            print("  No data recorded")
            return

        heights = [s["h"] for s in self.samples]
        pitches = [s["pitch"] for s in self.samples]
        rolls = [s["roll"] for s in self.samples]

        # Find walk start (first non-DSP phase after settling)
        walk_start = None
        for s in self.samples:
            if s["phase"] != "DSP" and walk_start is None:
                walk_start = s["t"]
                break

        print(f"\n  === DIAGNOSTICS REPORT ===")
        print(f"  Walk started:     t={walk_start:.1f}s" if walk_start else "  Walk never started")
        print(f"  Simulation end:   t={final_time:.1f}s")
        if self.fall_time:
            duration = self.fall_time - (walk_start or 0)
            print(f"  FELL at:          t={self.fall_time:.1f}s (walked {duration:.1f}s)")
            print(f"  Fall cause:       {self.fall_cause}")
        else:
            print(f"  Status:           UPRIGHT for entire test")

        print(f"\n  Height:  min={min(heights)*1000:.0f}mm  max={max(heights)*1000:.0f}mm  avg={np.mean(heights)*1000:.0f}mm")
        print(f"  Pitch:   min={np.degrees(min(pitches)):+.1f}deg  max={np.degrees(max(pitches)):+.1f}deg")
        print(f"  Roll:    min={np.degrees(min(rolls)):+.1f}deg  max={np.degrees(max(rolls)):+.1f}deg")

        # Phase distribution
        phases = [s["phase"] for s in self.samples]
        for p in ["DSP", "L_SWING", "R_SWING"]:
            count = phases.count(p)
            pct = count / len(phases) * 100
            if count > 0:
                print(f"  Phase {p:8s}:  {count} samples ({pct:.0f}%)")

        # Joint angle ranges
        print(f"\n  Joint angle ranges (during walk):")
        walk_samples = [s for s in self.samples if s["t"] >= (walk_start or 0)]
        if walk_samples:
            for key in ["l_hip_pitch_act", "r_hip_pitch_act",
                         "l_knee_act", "r_knee_act",
                         "l_ank_pitch_act", "r_ank_pitch_act"]:
                vals = [s[key] for s in walk_samples]
                print(f"    {key:25s}: [{np.degrees(min(vals)):+6.1f}, {np.degrees(max(vals)):+6.1f}] deg")

        # Height timeline (condensed, every 1s)
        print(f"\n  Timeline (every 0.5s):")
        print(f"  {'t':>5s}  {'Phase':<8s}  {'Height':>6s}  {'Pitch':>6s}  {'Roll':>6s}  {'X':>7s}")
        last_t = -0.5
        for s in self.samples:
            if s["t"] - last_t >= 0.5:
                last_t = s["t"]
                status = ""
                if self.fall_time and abs(s["t"] - self.fall_time) < 0.05:
                    status = " <-- FALL"
                print(
                    f"  {s['t']:5.1f}s {s['phase']:<8s} "
                    f"{s['h']*1000:5.0f}mm "
                    f"{np.degrees(s['pitch']):+5.1f}d "
                    f"{np.degrees(s['roll']):+5.1f}d "
                    f"{s['x']*1000:+6.0f}mm{status}"
                )

        passed = self.fall_time is None and min(heights) >= 0.200
        print(f"\n  Result: {'[PASS]' if passed else '[FAIL]'}")
        return passed


# ===== Run Modes =====

def run_headless(duration=20.0):
    """Run headless test with full diagnostics."""
    print("=" * 60)
    print("  ROBOTIS Walking Engine - Diagnostic Run")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    act_map = build_act_map(model)
    diag = WalkDiagnostics()

    engine = WalkingEngine()

    # Settle
    print("  Settling into standing pose...")
    standing = engine._standing_pose()
    for name, val in standing.items():
        if name in act_map:
            data.ctrl[act_map[name]] = val
    for _ in range(3000):
        mujoco.mj_step(model, data)

    h0 = body_height(model, data)
    print(f"  Standing height: {h0*1000:.0f}mm")
    print(f"  Sim timestep: {model.opt.timestep*1000:.1f}ms")

    engine.set_velocity(forward=0.025)
    engine.start()

    ctrl_dt = 1.0 / 50.0
    steps_per = int(ctrl_dt / model.opt.timestep)

    print(f"  Walking for {duration}s (ctrl @ {1/ctrl_dt:.0f}Hz, sim @ {1/model.opt.timestep:.0f}Hz)...")

    while data.time < duration + 6.0:  # +6 for settling time
        # Get IMU before control
        pitch, roll = get_imu(model, data)

        # Run walking engine
        angles = engine.update(ctrl_dt)
        WalkingEngine.apply_balance(angles, pitch, roll)

        # Apply to actuators
        for name, val in angles.items():
            if name in act_map:
                data.ctrl[act_map[name]] = val

        # Step simulation
        for _ in range(steps_per):
            mujoco.mj_step(model, data)

        # Record diagnostics
        h = body_height(model, data)
        pos = body_pos(model, data)
        pitch_after, roll_after = get_imu(model, data)
        diag.record(data.time, h, pitch_after, roll_after, engine.phase_name, angles, pos[0])

    passed = diag.print_report(data.time)
    return passed


def run_viewer():
    """Run with interactive MuJoCo viewer + diagnostics."""
    print("=" * 60)
    print("  OpenServoSim - ROBOTIS Walking (with diagnostics)")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    act_map = build_act_map(model)
    diag = WalkDiagnostics()

    engine = WalkingEngine()
    walk_started = False

    # Settle
    standing = engine._standing_pose()
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
        print(f"\n  [ERROR] Failed to create MuJoCo viewer window: {e}")
        print("  This is likely caused by 'Meta Virtual Monitor' (Quest headset driver)")
        print("  intercepting OpenGL. Try:")
        print("    1. Disable 'Meta Virtual Monitor' in Device Manager > Display adapters")
        print("    2. Or run:  python examples/03b_robotis_walk.py --headless --duration 10")
        sys.exit(1)

    with viewer_ctx as viewer:
        last_print = -2.0

        while viewer.is_running():
            t0 = time.time()
            pitch, roll = get_imu(model, data)

            if not walk_started and data.time > 2.0:
                walk_started = True
                engine.set_velocity(forward=0.025)
                engine.start()
                print("  Walking started!")

            if walk_started:
                angles = engine.update(ctrl_dt)
                WalkingEngine.apply_balance(angles, pitch, roll)
            else:
                angles = engine._standing_pose()

            for name, val in angles.items():
                if name in act_map:
                    data.ctrl[act_map[name]] = val

            for _ in range(steps_per):
                mujoco.mj_step(model, data)

            # Record diagnostics every step
            h = body_height(model, data)
            pos = body_pos(model, data)
            pitch_after, roll_after = get_imu(model, data)
            phase = engine.phase_name if walk_started else "SETTLE"
            diag.record(data.time, h, pitch_after, roll_after, phase, angles, pos[0])

            # Print status every 2s
            if data.time - last_print >= 2.0:
                last_print = data.time
                status = "WALKING" if h > 0.18 else "FALLEN!"
                print(
                    f"  t={data.time:5.1f}s  {phase:<8s}  "
                    f"h={h*1000:.0f}mm  x={pos[0]*1000:+.0f}mm  "
                    f"pitch={np.degrees(pitch_after):+.1f}d  roll={np.degrees(roll_after):+.1f}d  [{status}]"
                )
                # Stop printing after fall
                if h < 0.10:
                    break

            viewer.sync()
            sleep_time = ctrl_dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Print full report when viewer closes
    diag.print_report(data.time)


def main():
    parser = argparse.ArgumentParser(description="ROBOTIS Walking Engine Demo")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    if args.headless:
        success = run_headless(args.duration)
        sys.exit(0 if success else 1)
    else:
        run_viewer()


if __name__ == "__main__":
    main()
