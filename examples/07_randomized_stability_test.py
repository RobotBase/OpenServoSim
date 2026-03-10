#!/usr/bin/env python3
"""
OpenServoSim - Randomized Stability Search Pipeline

This script performs a hyperparameter sweep over walking gait and UVC balance
parameters. It simulates a rigorous "stress test" sequence of omnidirectional
movements in headless mode to find the most robust configuration that resists
drifting and falling during prolonged teleoperation.
"""

import os
import sys
import time
import math
import numpy as np
import mujoco
import json
from dataclasses import replace

import importlib.util

# Add project root to python path so we can import sim and controllers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.robotis_walking import WalkingParam, WalkingEngine
from controllers.uvc_walking import UVCParam, UVCWalkingEngine

# Dynamically import 05_uvc_walk.py
preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_uvc_walk.py")
spec = importlib.util.spec_from_file_location("walk_base", preset_path)
walk_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walk_base)

# =============================================================================
# Helper Functions
# =============================================================================
def get_imu(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    qw, qx, qy, qz = data.xquat[bid]
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    return pitch, roll

def body_height(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    return data.xpos[bid][2]

def body_pos(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    return data.xpos[bid].copy()

# =============================================================================
# Search Space Definition
# =============================================================================
def generate_random_params(base_preset, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    walk_p = base_preset["walk"]
    uvc_p = base_preset["uvc"]
    
    # Mutate parameters around the baseline (± percentage)
    new_walk = WalkingParam(
        # Fast cadence bounds
        period_time=np.random.uniform(0.400, 0.550),
        dsp_ratio=np.random.uniform(0.10, 0.25),
        
        # Step scaling bounds
        x_move_amplitude=min(walk_p.x_move_amplitude * np.random.uniform(0.8, 1.2), 0.045),
        z_move_amplitude=np.random.uniform(0.025, 0.040),
        y_swap_amplitude=np.random.uniform(0.015, 0.030),
        
        # Stance bounds
        init_z_offset=np.random.uniform(0.020, 0.050),
        pelvis_offset=np.random.uniform(math.radians(0.5), math.radians(2.0)),
        
        # Fast balance reflexes
        balance_ankle_pitch_gain=np.random.uniform(0.8, 1.5),
        balance_ankle_roll_gain=np.random.uniform(0.6, 1.2),
        balance_hip_roll_gain=np.random.uniform(0.3, 0.6),
        balance_knee_gain=np.random.uniform(0.3, 0.7),
        
        arm_swing_gain=np.random.uniform(0.8, 1.5)
    )
    
    new_uvc = UVCParam(
        gain_roll=np.random.uniform(0.20, 0.40),
        gain_pitch=np.random.uniform(0.15, 0.35),
        roll_scale=np.random.uniform(0.15, 0.30),
        pitch_scale=np.random.uniform(0.10, 0.25),
        roll_decay=np.random.uniform(0.85, 0.95),
        pitch_decay=np.random.uniform(0.85, 0.95),
        dead_zone=np.random.uniform(0.02, 0.06),
        max_correction=np.random.uniform(0.10, 0.20),
        warmup_time=1.0
    )
    
    return new_walk, new_uvc

# =============================================================================
# Headless Evaluation Function
# =============================================================================
def evaluate_params(walk_p, uvc_p):
    """
    Runs a 10-second headless simulation with a fixed sequence of movements.
    Returns a fitness score (higher is better) and metrics.
    """
    model = mujoco.MjModel.from_xml_path(walk_base.get_model_path())
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    act_map = walk_base.build_act_map(model)
    
    engine = UVCWalkingEngine(walk_p, uvc_p)
    ctrl_dt = 1.0 / 50.0  # 50Hz control loop
    steps_per_ctrl = int(ctrl_dt / model.opt.timestep)
    
    # 1. Settle
    standing = engine.walk._standing_pose()
    for name, val in standing.items():
        if name in act_map:
            data.ctrl[act_map[name]] = val
    for _ in range(1500):
        mujoco.mj_step(model, data)
        
    engine.start()
    
    fell = False
    min_h = 10.0
    
    stats = {
        "pitch": [],
        "roll": []
    }
    
    path_length = 0.0
    prev_pos = body_pos(model, data)
    
    # 2. Main Simulation Loop (10 seconds total)
    total_time = 10.0
    for tick in range(int(total_time / ctrl_dt)):
        t = tick * ctrl_dt
        
        # Teleop Stress Test Sequence
        tgt_fwd = 0.0
        tgt_lat = 0.0
        tgt_turn = 0.0
        
        if 2.0 <= t < 4.0:
            # Forward + Turn Left arc
            tgt_fwd = walk_p.x_move_amplitude
            tgt_turn = 0.15
        elif 4.0 <= t < 6.0:
            # Backward + Sidestep Right
            tgt_fwd = -walk_p.x_move_amplitude * 0.7
            tgt_lat = -0.015
        elif 6.0 <= t < 8.0:
            # Sudden Forward
            tgt_fwd = walk_p.x_move_amplitude
        elif 8.0 <= t < 10.0:
            # Crazy spin
            tgt_turn = -0.25
            
        # Hard setting velocity (instant acceleration like keyboard)
        engine.set_velocity(tgt_fwd, tgt_lat, tgt_turn)
        
        # Step control
        pitch, roll = get_imu(model, data)
        angles = engine.update(ctrl_dt, pitch, roll)
        WalkingEngine.apply_balance(angles, pitch, roll)
        
        for name, val in angles.items():
            if name in act_map:
                data.ctrl[act_map[name]] = val
                
        # Step physics
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)
            
        # Collect Metrics
        h = body_height(model, data)
        if h < 0.17:
            fell = True
            break
            
        min_h = min(min_h, h)
        stats["pitch"].append(math.degrees(pitch))
        stats["roll"].append(math.degrees(roll))
        
        cur_pos = body_pos(model, data)
        dx = cur_pos[0] - prev_pos[0]
        dy = cur_pos[1] - prev_pos[1]
        if t > 2.0:  # Only measure movement path after settling
            path_length += math.sqrt(dx**2 + dy**2)
        prev_pos = cur_pos
        
    # 3. Calculate Fitness
    if fell:
        # Score is purely based on how long it survived (t)
        score = float(t)
    else:
        # Penalize variance in pitch/roll (instability)
        pitch_var = np.var(stats["pitch"])
        roll_var = np.var(stats["roll"])
        
        # Reward maintaining height
        height_reward = min_h * 10.0
        
        # Reward actual movement distance (avoids parameters that just stand in place safely)
        dist_reward = path_length * 20.0
        
        # Base score of 10.0 for surviving the full 10 seconds + bonus
        score = 10.0 + (dist_reward + height_reward) / (1.0 + pitch_var*0.1 + roll_var*0.1)
        
    return score, fell, path_length, min_h

# =============================================================================
# Main Sweep Script
# =============================================================================
def main():
    print("="*60)
    print(" Phase 6: Randomized Stability Search Pipeline ")
    print("="*60)
    
    NUM_ITERS = 100
    base = walk_base.SPEED_PRESETS["ninja"]
    
    best_score = -1.0
    best_params = None
    best_idx = -1
    
    results = []
    
    start_time = time.time()
    
    for i in range(NUM_ITERS):
        sys.stdout.write(f"\rTesting {i+1}/{NUM_ITERS} ... ")
        sys.stdout.flush()
        
        params_w, params_u = generate_random_params(base)
        score, fell, dist, min_h = evaluate_params(params_w, params_u)
        
        # Store barebones copy of params for JSON logging
        p_dict = {
            "iter": i+1,
            "score": score,
            "fell": fell,
            "path_length": dist,
            "period_time": params_w.period_time,
            "dsp_ratio": params_w.dsp_ratio,
            "balance_ankle_pitch_gain": params_w.balance_ankle_pitch_gain,
            "uvc_gain_roll": params_u.gain_roll,
            "uvc_gain_pitch": params_u.gain_pitch
        }
        results.append(p_dict)
        
        if score > best_score and not fell:
            best_score = score
            best_params = (params_w, params_u)
            best_idx = i + 1
            print(f"\n   -> New BEST! Score: {score:.2f} (Dist: {dist*1000:.0f}mm, MinH: {min_h*1000:.0f}mm)")
            
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Sweep completed in {total_time:.1f} seconds.")
    print(f"Top Performer: Iteration #{best_idx} with Score {best_score:.2f}")
    
    if best_params is not None:
        print("\nOptimal Fast Params Found (Copy to OP3 Presets):")
        w, u = best_params
        
        print(f"    \"walk\": WalkingParam(")
        print(f"        init_z_offset={w.init_z_offset:.3f},")
        print(f"        x_move_amplitude={w.x_move_amplitude:.3f},")
        print(f"        period_time={w.period_time:.3f},")
        print(f"        z_move_amplitude={w.z_move_amplitude:.3f},")
        print(f"        dsp_ratio={w.dsp_ratio:.3f},")
        print(f"        y_swap_amplitude={w.y_swap_amplitude:.3f},")
        print(f"        z_swap_amplitude={w.z_swap_amplitude:.3f},")
        print(f"        arm_swing_gain={w.arm_swing_gain:.2f},")
        print(f"        pelvis_offset={w.pelvis_offset:.3f},")
        print(f"        balance_ankle_pitch_gain={w.balance_ankle_pitch_gain:.2f},")
        print(f"        balance_ankle_roll_gain={w.balance_ankle_roll_gain:.2f},")
        print(f"        balance_hip_roll_gain={w.balance_hip_roll_gain:.2f},")
        print(f"        balance_knee_gain={w.balance_knee_gain:.2f},")
        print(f"    ),")
        
        print(f"    \"uvc\": UVCParam(")
        print(f"        gain_roll={u.gain_roll:.2f}, gain_pitch={u.gain_pitch:.2f},")
        print(f"        dead_zone={u.dead_zone:.3f},")
        print(f"        roll_scale={u.roll_scale:.2f}, pitch_scale={u.pitch_scale:.2f},")
        print(f"        max_correction={u.max_correction:.2f},")
        print(f"        roll_decay={u.roll_decay:.2f}, pitch_decay={u.pitch_decay:.2f},")
        print(f"        warmup_time=1.0,")
        print(f"    ),")
        
    # Output JSON log
    log_file = "stability_sweep_results.json"
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDataset logged to {log_file}")

if __name__ == "__main__":
    main()
