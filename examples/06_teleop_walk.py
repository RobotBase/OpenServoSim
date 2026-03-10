"""
OpenServoSim - UVC Omnidirectional Teleoperation Demo

Controls (Avoids MuJoCo Camera Conflicts):
  [I] / [K] or Numpad [8] / [2] : Forward / Backward
  [J] / [L] or Numpad [4] / [6] : Sidestep Left / Right
  [U] / [O] or Numpad [7] / [9] : Turn Left / Right
  [Space] or Numpad [5]         : Stop all movement
  
Note: Requires 'keyboard' module (`pip install keyboard`)
      Run as Administrator/root if keyboard module complains about permissions.
"""

import os
import sys
import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import argparse
import importlib.util

# Add project root to python path so we can import sim and controllers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import keyboard
except ImportError:
    print("\n[ERROR] 'keyboard' module not found.")
    print("Please install it: pip install keyboard")
    print("Note: On Windows/Linux, this may require Administrator/sudo privileges to run.\n")
    exit(1)

from controllers.uvc_walking import UVCWalkingEngine

# Dynamically import SPEED_PRESETS from 05_uvc_walk.py
preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_uvc_walk.py")
spec = importlib.util.spec_from_file_location("uvc_walk", preset_path)
uvc_walk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uvc_walk)
SPEED_PRESETS = uvc_walk.SPEED_PRESETS

# Helper functions for the OP3 model
def get_model_path():
    pr = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(pr, "models", "reference", "robotis_op3", "scene_enhanced.xml")
    return p if os.path.exists(p) else os.path.join(
        pr, "models", "reference", "robotis_op3", "scene.xml")

def get_imu(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    qw, qx, qy, qz = data.xquat[bid]
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    return pitch, roll

def build_act_map(model):
    return {model.actuator(i).name: i for i in range(model.nu)}

# Max speeds
MAX_FWD = 0.040   # 40mm/step
MAX_BWD = -0.030  # -30mm/step
MAX_LAT = 0.020   # 20mm/step sidestep
MAX_TURN = math.radians(10.0)  # 10 degrees/step turn

# Acceleration (units per second)
ACCEL_FWD = 0.400  # Reaches max in 0.1s
ACCEL_LAT = 0.200  # Reaches max in 0.1s
ACCEL_TURN = math.radians(100.0)

def run_teleop():
    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    act_map = build_act_map(model)
    
    # Use the 'robust' preset discovered by our random search pipeline!
    preset = SPEED_PRESETS["robust"]
    engine = UVCWalkingEngine(preset["walk"], preset["uvc"])
    
    # Current velocity states
    cur_fwd = 0.0
    cur_lat = 0.0
    cur_turn = 0.0
    
    print("\n" + "="*50)
    print(" 🎮 UVC Teleoperation Ready 🎮")
    print("="*50)
    print("  Controls (Optimized to avoid MuJoCo camera conflicts):")
    print("    [I] or Numpad [8] : Forward")
    print("    [K] or Numpad [2] : Backward")
    print("    [J] or Numpad [4] : Sidestep Left (Crab walk)")
    print("    [L] or Numpad [6] : Sidestep Right")
    print("    [U] or Numpad [7] : Turn Left in place")
    print("    [O] or Numpad [9] : Turn Right in place")
    print("    [SPACE] or Np [5] : Stop instantly")
    print("="*50 + "\n")

    # Physics timing
    ctrl_dt = 1.0 / 50.0  # 50 Hz control loop (20ms)
    steps_per_ctrl = int(ctrl_dt / model.opt.timestep)  # 20ms / 2ms = 10 steps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Give physics a moment to settle
        # Initial pose
        standing = engine.walk._standing_pose()
        for name, val in standing.items():
            if name in act_map:
                data.ctrl[act_map[name]] = val
                
        for _ in range(3000): # Allow to drop and stabilize
            mujoco.mj_step(model, data)
            
        engine.start()
        
        last_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- 1. Keyboard Input Polling ---
            tgt_fwd = 0.0
            tgt_lat = 0.0
            tgt_turn = 0.0
            
            if not (keyboard.is_pressed('space') or keyboard.is_pressed('5')):
                if keyboard.is_pressed('i') or keyboard.is_pressed('8') or keyboard.is_pressed('up'): tgt_fwd = MAX_FWD
                if keyboard.is_pressed('k') or keyboard.is_pressed('2') or keyboard.is_pressed('down'): tgt_fwd = MAX_BWD
                
                if keyboard.is_pressed('j') or keyboard.is_pressed('4') or keyboard.is_pressed('left'): tgt_lat = MAX_LAT
                if keyboard.is_pressed('l') or keyboard.is_pressed('6') or keyboard.is_pressed('right'): tgt_lat = -MAX_LAT
                
                if keyboard.is_pressed('u') or keyboard.is_pressed('7') or keyboard.is_pressed(','): tgt_turn = MAX_TURN
                if keyboard.is_pressed('o') or keyboard.is_pressed('9') or keyboard.is_pressed('.'): tgt_turn = -MAX_TURN
            
            # --- 2. Smooth Acceleration Interpolation ---
            # Forward/Backward
            if cur_fwd < tgt_fwd:
                cur_fwd = min(cur_fwd + ACCEL_FWD * ctrl_dt, tgt_fwd)
            elif cur_fwd > tgt_fwd:
                cur_fwd = max(cur_fwd - ACCEL_FWD * ctrl_dt, tgt_fwd)
                
            # Lateral
            if cur_lat < tgt_lat:
                cur_lat = min(cur_lat + ACCEL_LAT * ctrl_dt, tgt_lat)
            elif cur_lat > tgt_lat:
                cur_lat = max(cur_lat - ACCEL_LAT * ctrl_dt, tgt_lat)
                
            # Turn
            if cur_turn < tgt_turn:
                cur_turn = min(cur_turn + ACCEL_TURN * ctrl_dt, tgt_turn)
            elif cur_turn > tgt_turn:
                cur_turn = max(cur_turn - ACCEL_TURN * ctrl_dt, tgt_turn)
                
            # Apply velocities
            engine.set_velocity(forward=cur_fwd, lateral=cur_lat, turn=cur_turn)
            
            # --- 3. Run Control & Physics ---
            pitch, roll = get_imu(model, data)
            target_angles = engine.update(ctrl_dt, pitch, roll)
            
            # Apply secondary fast ankle balance logic
            from controllers.robotis_walking import WalkingEngine
            WalkingEngine.apply_balance(target_angles, pitch, roll)
            
            for name, val in target_angles.items():
                if name in act_map:
                    data.ctrl[act_map[name]] = val
            
            # Advance physics engine accurately
            for _ in range(steps_per_ctrl):
                mujoco.mj_step(model, data)
                
            viewer.sync()
            
            # Timekeeping (50Hz control loop)
            # Wait for next control tick to match real time
            elapsed = time.time() - step_start
            time_until_next_step = ctrl_dt - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run_teleop()
