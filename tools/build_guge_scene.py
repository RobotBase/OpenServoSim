#!/usr/bin/env python3
"""
Build the correct Guge MuJoCo scene from URDF using MuJoCo's native compiler.

Steps:
1. Use MuJoCo's native URDF loader (handles RPY→quat correctly)
2. Save the compiled MJCF
3. Programmatically add: actuators, scene, mount stand, sensors, cameras
4. Validate the model
5. Render screenshots from multiple angles for visual verification

Usage:
    MUJOCO_GL=egl /home/zero/mujoco_playground/.venv/bin/python3 tools/build_guge_scene.py
"""

import os
import sys
import shutil
import numpy as np
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import mujoco

URDF_PATH = os.path.join(PROJECT_ROOT, "models", "guge_urdf", "urdf", "guge_urdf.urdf")
MESH_DIR = os.path.join(PROJECT_ROOT, "models", "guge_urdf", "meshes")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "guge")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "scene_guge.xml")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")


def step1_compile_urdf():
    """Use MuJoCo's native URDF compiler."""
    print("Step 1: Native URDF compilation...")

    # Copy meshes to temp dir alongside URDF
    tmp_dir = "/tmp/guge_compile"
    os.makedirs(tmp_dir, exist_ok=True)
    for f in os.listdir(MESH_DIR):
        if f.upper().endswith(".STL"):
            shutil.copy2(os.path.join(MESH_DIR, f), os.path.join(tmp_dir, f))

    # Fix mesh paths in URDF
    with open(URDF_PATH, 'r') as f:
        urdf = f.read()
    urdf = urdf.replace('package://guge_urdf/meshes/', '')

    tmp_urdf = os.path.join(tmp_dir, "guge.urdf")
    with open(tmp_urdf, 'w') as f:
        f.write(urdf)

    # Load and save via MuJoCo
    model = mujoco.MjModel.from_xml_path(tmp_urdf)
    compiled_path = os.path.join(tmp_dir, "guge_compiled.xml")
    mujoco.mj_saveLastXML(compiled_path, model)

    print(f"  Compiled: nq={model.nq}, nv={model.nv}, njnt={model.njnt}, nbody={model.nbody}")
    return compiled_path


def step2_build_scene(compiled_path):
    """Add scene elements to the compiled MJCF."""
    print("Step 2: Building scene XML...")

    tree = ET.parse(compiled_path)
    root = tree.getroot()

    # Update model name
    root.set("model", "guge_upper_body")

    # --- Compiler: set meshdir relative to output location ---
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", "../guge_urdf/meshes/")

    # --- Option ---
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicit")

    # --- Visual ---
    visual = ET.SubElement(root, "visual")
    glob = ET.SubElement(visual, "global")
    glob.set("offwidth", "1920")
    glob.set("offheight", "1080")

    # --- Default ---
    default = ET.SubElement(root, "default")
    jnt_def = ET.SubElement(default, "joint")
    jnt_def.set("damping", "0.5")
    jnt_def.set("armature", "0.01")

    # --- Assets: add textures and materials ---
    asset = root.find("asset")

    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0",
        "width": "512", "height": "3072"
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker",
        "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"
    })

    # Remove content_type from mesh elements (not needed with meshdir)
    for mesh_elem in asset.findall("mesh"):
        if "content_type" in mesh_elem.attrib:
            del mesh_elem.attrib["content_type"]

    # --- Worldbody: wrap robot in mount stand ---
    worldbody = root.find("worldbody")

    # Save original robot body tree
    robot_elements = list(worldbody)

    # Clear worldbody
    for elem in robot_elements:
        worldbody.remove(elem)

    # Add ground
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "5 5 0.1",
        "material": "groundplane"
    })

    # Add lights
    ET.SubElement(worldbody, "light", {
        "name": "spotlight", "mode": "targetbodycom", "target": "spine3",
        "pos": "0 -1 2", "diffuse": "0.8 0.8 0.8"
    })
    ET.SubElement(worldbody, "light", {
        "name": "ambient", "directional": "true",
        "pos": "0 0 3", "dir": "0 0 -1", "diffuse": "0.4 0.4 0.4"
    })

    # Mount stand body at height 0.8m
    stand = ET.SubElement(worldbody, "body", {"name": "stand", "pos": "0 0 0.8"})
    ET.SubElement(stand, "geom", {
        "name": "stand_pole", "type": "cylinder",
        "size": "0.02 0.4", "pos": "0 0 -0.4",
        "rgba": "0.3 0.3 0.3 1", "contype": "0", "conaffinity": "0"
    })
    ET.SubElement(stand, "geom", {
        "name": "stand_base", "type": "cylinder",
        "size": "0.15 0.02", "pos": "0 0 -0.8",
        "rgba": "0.3 0.3 0.3 1", "contype": "0", "conaffinity": "0"
    })

    # Create base_link body inside stand (fixed)
    base_body = ET.SubElement(stand, "body", {
        "name": "base_link", "pos": "0 0 0"
    })

    # Attach original robot geom and children to base_link
    for elem in robot_elements:
        if elem.tag == "geom":
            # This is the base_link geom
            base_body.append(elem)
        elif elem.tag == "body":
            # This is spine1 and children
            base_body.append(elem)

    # Add IMU site to spine3
    spine3 = None
    for body in base_body.iter("body"):
        if body.get("name") == "spine3":
            spine3 = body
            break
    if spine3 is not None:
        ET.SubElement(spine3, "site", {
            "name": "imu_site", "pos": "0 0 0", "size": "0.005"
        })

    # --- Actuators ---
    actuator_elem = ET.SubElement(root, "actuator")
    joint_names = [
        "spine1", "spine2", "spine3",
        "left_arm1", "left_arm2", "left_arm3", "left_arm4", "left_arm5",
        "right_arm1", "right_arm2", "right_arm3", "right_arm4", "right_arm5",
    ]

    # Get joint ranges from compiled model
    tmp_model = mujoco.MjModel.from_xml_path(compiled_path)
    for jname in joint_names:
        jid = tmp_model.joint(jname).id
        lower = tmp_model.jnt_range[jid, 0]
        upper = tmp_model.jnt_range[jid, 1]
        ET.SubElement(actuator_elem, "position", {
            "name": f"{jname}_motor",
            "joint": jname,
            "kp": "50",
            "ctrlrange": f"{lower:.4f} {upper:.4f}",
        })

    # --- Sensors ---
    sensor_elem = ET.SubElement(root, "sensor")
    ET.SubElement(sensor_elem, "gyro", {"name": "imu_gyro", "site": "imu_site"})
    ET.SubElement(sensor_elem, "accelerometer", {"name": "imu_accel", "site": "imu_site"})

    # --- Write ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pretty-print XML
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_PATH, xml_declaration=True, encoding="unicode")

    print(f"  Written to: {OUTPUT_PATH}")
    return OUTPUT_PATH


def step3_validate(scene_path):
    """Load and validate the final model."""
    print("Step 3: Validating model...")

    model = mujoco.MjModel.from_xml_path(scene_path)
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")

    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print(f"  Simulation step OK")

    print(f"\n  Joints:")
    for i in range(model.njnt):
        j = model.joint(i)
        lo = np.degrees(model.jnt_range[i, 0])
        hi = np.degrees(model.jnt_range[i, 1])
        print(f"    {i:2d}: {j.name:<15s}  [{lo:+7.1f}°, {hi:+7.1f}°]")

    print(f"\n  Actuators: {model.nu}")
    for i in range(model.nu):
        print(f"    {i:2d}: {model.actuator(i).name}")

    return model


def step4_render_verification(model):
    """Render screenshots from multiple angles + per-joint video."""
    print("\nStep 4: Rendering verification...")
    os.makedirs(VIDEO_DIR, exist_ok=True)

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)

    # First: render neutral pose from multiple angles
    mujoco.mj_forward(model, data)

    angles = [
        ("front", 180, -10),
        ("left",  90,  -10),
        ("right", 270, -10),
        ("back",    0, -10),
        ("top",   180, -80),
    ]

    for name, azimuth, elevation in angles:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.85]
        camera.distance = 1.2
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera)
        frame = renderer.render()

        img_path = os.path.join(VIDEO_DIR, f"neutral_{name}.png")
        from PIL import Image
        Image.fromarray(frame).save(img_path)
        print(f"  Saved: {img_path}")

    # Second: render per-joint sweep video
    print("\n  Rendering joint verification video...")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.85]
    camera.distance = 1.2
    camera.azimuth = 180
    camera.elevation = -10

    fps = 30
    dt = model.opt.timestep
    steps_per_frame = int(1.0 / fps / dt)
    seconds_per_joint = 4.0
    frames_per_joint = int(seconds_per_joint * fps)

    frames = []
    for ji in range(model.njnt):
        jname = model.joint(ji).name
        lower = model.jnt_range[ji, 0]
        upper = model.jnt_range[ji, 1]
        amp = (upper - lower) / 2.0 * 0.7

        print(f"    [{ji+1:2d}/{model.njnt}] {jname}")

        # Reset all joints to neutral
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)

        for fi in range(frames_per_joint):
            t = fi / frames_per_joint
            angle = amp * np.sin(2 * np.pi * t)

            data.ctrl[:] = 0
            data.ctrl[ji] = angle

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera)
            frames.append(renderer.render().copy())

    # Return to rest
    data.ctrl[:] = 0
    for _ in range(fps):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())

    renderer.close()

    video_path = os.path.join(VIDEO_DIR, "joint_verification.mp4")
    try:
        import mediapy as media
        media.write_video(video_path, frames, fps=fps)
        print(f"  Video: {video_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    except ImportError:
        print("  [mediapy not available]")

    return video_path


def main():
    print("=" * 60)
    print("  Guge Scene Builder (Native URDF Compiler)")
    print("=" * 60)

    compiled = step1_compile_urdf()
    scene = step2_build_scene(compiled)
    model = step3_validate(scene)
    video = step4_render_verification(model)

    print("\n" + "=" * 60)
    print("  Build complete!")
    print(f"  Scene: {scene}")
    print(f"  Video: {video}")
    print("=" * 60)


if __name__ == "__main__":
    main()
