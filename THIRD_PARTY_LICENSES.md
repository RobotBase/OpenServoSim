# Third-Party Licenses

This file documents the licenses and attribution for third-party robot models
included in the `models/reference/` directory.

**These models are included for reference and comparison purposes.**
Each model retains its original license. Please refer to the individual
LICENSE files within each model directory for the complete terms.

---

## Robotis OP3 (MuJoCo Menagerie)

- **Source**: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotis_op3)
- **License**: Apache License 2.0
- **Original Author**: Google DeepMind (MJCF conversion), Robotis (robot design)
- **Local Path**: `models/reference/robotis_op3/`
- **Notes**: Complete MuJoCo MJCF model with assets. Used by DeepMind for OP3 Soccer research (agile skills, sim-to-real transfer).

## DARwin-OP

- **Source**: [HumaRobotics/darwin_description](https://github.com/HumaRobotics/darwin_description)
- **License**: BSD 3-Clause License
- **Original Author**: HumaRobotics
- **Local Path**: `models/reference/darwin_op/`
- **Notes**: ROS URDF description with STL meshes. 20-DOF humanoid, Dynamixel servos.

## ToddlerBot

- **Source**: [hshi74/toddlerbot](https://github.com/hshi74/toddlerbot)
- **License**: MIT License
- **Original Author**: Haoran Shi et al. (Stanford)
- **Local Path**: `models/reference/toddlerbot/`
- **Notes**: Multiple variants (2xc, 2xm, with/without grippers) in both URDF and MuJoCo XML format. Includes MJX-compatible versions for GPU-accelerated simulation.

## Poppy Humanoid

- **Source**: [poppy-project/poppy-humanoid](https://github.com/poppy-project/poppy-humanoid)
- **License**: GPL v3 (software), CC-BY-SA (hardware)
- **Original Author**: Poppy Project / INRIA Flowers Lab
- **Local Path**: `models/reference/poppy_humanoid/`
- **Notes**: Educational humanoid robot with ~25 DOF. URDF with DAE meshes. Note: GPL v3 license applies to software components.

> **⚠️ GPL Notice**: The Poppy Humanoid software components use GPL v3, which has
> copyleft requirements. The model files are included here for reference and
> study purposes. If you redistribute derivative works based on these model
> files, you must comply with GPL v3 terms.
