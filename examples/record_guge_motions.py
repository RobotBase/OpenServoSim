#!/usr/bin/env python3
"""
Guge Motion Recorder — Render all 12 motions as MP4 videos.

Renders each motion from the MotionLibrary through MuJoCo simulation
with physics-accurate playback and smooth camera views.

Usage:
    MUJOCO_GL=egl /home/zero/mujoco_playground/.venv/bin/python3 \
        examples/record_guge_motions.py

Output:
    models/guge/videos/motion_01_wave_hello.mp4
    models/guge/videos/motion_02_bow.mp4
    ...
    models/guge/videos/motion_all_demo.mp4 (all motions concatenated)
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import mujoco
from controllers.motion_library import MotionLibrary

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "guge", "scene_guge.xml")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "models", "guge", "videos")

FPS = 30
RESOLUTION = (1280, 720)


def render_motion(model, motion_name, lib, camera):
    """Render a single motion and return frames."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=RESOLUTION[1], width=RESOLUTION[0])

    motion = lib.get(motion_name)
    duration = lib.get_duration(motion)
    dt = model.opt.timestep
    steps_per_frame = int(1.0 / FPS / dt)

    # Add 0.5s padding at start and end
    total_time = duration + 1.0
    num_frames = int(total_time * FPS)

    frames = []
    sim_time = 0.0

    for frame_i in range(num_frames):
        t = frame_i / FPS - 0.5  # Start 0.5s before motion

        if 0 <= t <= duration:
            angles = lib.evaluate(motion, t)
            data.ctrl[:] = angles
        else:
            data.ctrl[:] = 0

        # Simulate
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
            sim_time += dt

        # Render
        renderer.update_scene(data, camera)
        frame = renderer.render()
        frames.append(frame.copy())

    renderer.close()
    return frames


def main():
    print("=" * 60)
    print("  Guge Motion Recorder")
    print("=" * 60)

    os.makedirs(VIDEO_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    lib = MotionLibrary()

    # Camera setup
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.85]
    camera.distance = 1.0
    camera.azimuth = 180
    camera.elevation = -10

    motion_names = lib.list_motions()
    all_frames = []

    for idx, name in enumerate(motion_names, 1):
        motion = lib.get(name)
        duration = lib.get_duration(motion)
        print(f"\n  [{idx:2d}/{len(motion_names)}] {name} ({duration:.1f}s) - {motion.description}")

        frames = render_motion(model, name, lib, camera)
        all_frames.extend(frames)

        # Save individual video
        video_path = os.path.join(VIDEO_DIR, f"motion_{idx:02d}_{name}.mp4")
        try:
            import mediapy as media
            media.write_video(video_path, frames, fps=FPS)
            print(f"         → {os.path.basename(video_path)} ({len(frames)} frames)")
        except ImportError:
            print(f"         [mediapy not available, skipping individual save]")

    # Save combined demo video
    print(f"\n  Saving combined demo ({len(all_frames)} frames)...")
    demo_path = os.path.join(VIDEO_DIR, "motion_all_demo.mp4")
    try:
        import mediapy as media
        media.write_video(demo_path, all_frames, fps=FPS)
        print(f"  → {demo_path}")
        total_duration = len(all_frames) / FPS
        print(f"  Total: {total_duration:.1f}s, {len(all_frames)} frames")
    except ImportError:
        print("  [mediapy not available]")

    print(f"\n  All videos saved to: {VIDEO_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
