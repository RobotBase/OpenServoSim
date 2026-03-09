# Reference Models

This directory contains model files from other open-source servo-based humanoid robot projects.
These are included for **reference, comparison, and algorithm development** purposes.

> **⚠️ Important**: Each model has its own license. See [THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md) for details.

## Models Overview

| Model | DOF | Servos | Format | License | Source |
|-------|-----|--------|--------|---------|--------|
| **Robotis OP3** | 20 | Dynamixel | MJCF (MuJoCo) | Apache 2.0 | [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotis_op3) |
| **DARwin-OP** | 20 | Dynamixel | URDF (ROS) | BSD-3 | [darwin_description](https://github.com/HumaRobotics/darwin_description) |
| **ToddlerBot** | varies | Dynamixel XC/XM | URDF + MJCF | MIT | [toddlerbot](https://github.com/hshi74/toddlerbot) |
| **Poppy Humanoid** | ~25 | Dynamixel MX | URDF (ROS) | GPL v3 | [poppy-humanoid](https://github.com/poppy-project/poppy-humanoid) |

## Usage

### Load Robotis OP3 in MuJoCo
```python
import mujoco
model = mujoco.MjModel.from_xml_path("models/reference/robotis_op3/scene.xml")
```

### Load DARwin-OP URDF
```python
# Requires a URDF parser (e.g., mujoco's compile or pybullet)
import mujoco
model = mujoco.MjModel.from_xml_path("models/reference/darwin_op/urdf/darwin.urdf")
```

### Load ToddlerBot in MuJoCo
```python
import mujoco
model = mujoco.MjModel.from_xml_path("models/reference/toddlerbot/toddlerbot_2xc/scene.xml")
```

## Why These Models?

All these robots share a key characteristic with the OpenServoSim project:
**they are driven by serial bus servos** (Dynamixel series), not BLDC motors.
This makes their kinematic structures, control approaches, and physical
constraints directly relevant for developing and testing servo-specific
control algorithms.
