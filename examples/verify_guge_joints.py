#!/usr/bin/env python3
"""
Guge Joint Verification & Visual Test

Tests each joint individually and renders a video showing all joints
moving in sequence. This verifies:
1. Each joint moves in the expected direction
2. Mesh rendering is correct
3. Joint limits are respected

Usage:
    # Using mujoco_playground venv:
    /home/zero/mujoco_playground/.venv/bin/python3 examples/verify_guge_joints.py

Output:
    models/guge/videos/joint_verification.mp4
"""

import os
import sys
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import mujoco

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "guge", "scene_guge.xml")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "models", "guge", "videos")

# Joint descriptions for annotation
JOINT_INFO = {
    "spine1":     "腰部旋转 (Waist Rotation)",
    "spine2":     "腰部前后 (Waist Pitch)",
    "spine3":     "躯干侧倾 (Torso Roll)",
    "left_arm1":  "左肩外展 (L Shoulder Abduction)",
    "left_arm2":  "左肩前屈 (L Shoulder Flexion)",
    "left_arm3":  "左上臂旋转 (L Upper Arm Rotation)",
    "left_arm4":  "左肘屈伸 (L Elbow Flexion)",
    "left_arm5":  "左腕旋转 (L Wrist Rotation)",
    "right_arm1": "右肩外展 (R Shoulder Abduction)",
    "right_arm2": "右肩前屈 (R Shoulder Flexion)",
    "right_arm3": "右上臂旋转 (R Upper Arm Rotation)",
    "right_arm4": "右肘屈伸 (R Elbow Flexion)",
    "right_arm5": "右腕旋转 (R Wrist Rotation)",
}


def verify_joints():
    """Test each joint individually and render video."""
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"Model: nq={model.nq}, nu={model.nu}, njnt={model.njnt}")

    # Get joint names and indices
    joint_names = []
    for i in range(model.njnt):
        name = model.joint(i).name
        joint_names.append(name)
        info = JOINT_INFO.get(name, "Unknown")
        lower = np.degrees(model.jnt_range[i, 0])
        upper = np.degrees(model.jnt_range[i, 1])
        print(f"  Joint {i:2d}: {name:<15s} [{lower:+7.1f}°, {upper:+7.1f}°] - {info}")

    # Setup renderer
    os.makedirs(VIDEO_DIR, exist_ok=True)
    renderer = mujoco.Renderer(model, height=720, width=1280)

    # Camera setup - front view
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.8]  # Look at stand height
    camera.distance = 1.2
    camera.azimuth = 180     # Front view
    camera.elevation = -15   # Slightly above

    print("\nRendering joint verification video...")

    frames = []
    fps = 30
    dt = model.opt.timestep

    # For each joint: hold neutral (1s) → sweep positive (1.5s) → sweep negative (1.5s) → return (1s)
    seconds_per_joint = 4.0
    frames_per_joint = int(seconds_per_joint * fps)
    steps_per_frame = int(1.0 / fps / dt)

    for ji, jname in enumerate(joint_names):
        info = JOINT_INFO.get(jname, "")
        lower = model.jnt_range[ji, 0]
        upper = model.jnt_range[ji, 1]
        mid = (lower + upper) / 2.0
        amp = (upper - lower) / 2.0 * 0.7  # 70% of range for safety

        print(f"  [{ji+1:2d}/{len(joint_names)}] {jname}: {info}")

        for fi in range(frames_per_joint):
            t = fi / frames_per_joint  # 0..1

            # Trajectory: 0→peak→-peak→0 (sinusoidal)
            angle = amp * np.sin(2 * np.pi * t)

            # Reset all controls to zero, set only this joint
            data.ctrl[:] = 0
            data.ctrl[ji] = angle

            # Simulate
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            # Render
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame.copy())

    # Return to rest for 1 second
    data.ctrl[:] = 0
    for fi in range(fps):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())

    renderer.close()

    # Save video
    video_path = os.path.join(VIDEO_DIR, "joint_verification.mp4")
    try:
        import mediapy as media
        media.write_video(video_path, frames, fps=fps)
        print(f"\n  Video saved: {video_path}")
    except ImportError:
        # Fallback: save as individual frames
        print("  mediapy not available, saving frames as images...")
        import struct
        frame_dir = os.path.join(VIDEO_DIR, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(os.path.join(frame_dir, f"frame_{i:04d}.png"))
        print(f"  Frames saved to: {frame_dir}/")
        # Try ffmpeg
        os.system(f"ffmpeg -y -r {fps} -i {frame_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path} 2>/dev/null")
        if os.path.exists(video_path):
            print(f"  Video saved: {video_path}")

    print(f"\n  Total frames: {len(frames)}")
    print(f"  Duration: {len(frames)/fps:.1f}s")
    print(f"  Each joint gets {seconds_per_joint:.1f}s of motion")

    return video_path


if __name__ == "__main__":
    verify_joints()
