---
description: How to verify and debug MuJoCo joint axes for ROBOTIS OP3 (and similar humanoid robots)
---

# MuJoCo Joint Axis Verification for OP3

Hard-won lessons from debugging joint directions on the ROBOTIS OP3 MuJoCo model. These patterns apply to any humanoid robot model.

## Key Concept: ROBOTIS `getJointDirection()`

ROBOTIS uses a simple convention to compute a joint's "direction multiplier":

```
direction = axis.x + axis.y + axis.z
```

This gives `+1` or `-1`. The walking module multiplies all joint commands by this direction. The source file is `op3_kinematics_dynamics.cpp`.

## OP3 Joint Axis Reference (All 20 Joints Verified)

All axes match between the MuJoCo model (`op3.xml`) and ROBOTIS source:

| Joint | Axis | Dir | Notes |
|-------|------|-----|-------|
| head_pan | (0, 0, +1) | +1 | Z+ yaw |
| head_tilt | (0, -1, 0) | -1 | Y- pitch |
| **l_sho_pitch** | **(0, +1, 0)** | **+1** | Y+ = positive swings arm forward |
| **r_sho_pitch** | **(0, -1, 0)** | **-1** | Y- = positive swings arm **backward** (mirrored!) |
| l_sho_roll | (-1, 0, 0) | -1 | X- = +1.3 brings arm down |
| r_sho_roll | (-1, 0, 0) | -1 | X- = -1.3 brings arm down |
| **l_el** | **(+1, 0, 0)** | **+1** | X+ = negative bends forearm inward |
| **r_el** | **(+1, 0, 0)** | **+1** | X+ = positive bends forearm inward |
| l_hip_yaw | (0, 0, -1) | -1 | |
| l_hip_roll | (-1, 0, 0) | -1 | |
| l_hip_pitch | (0, +1, 0) | +1 | |
| l_knee | (0, +1, 0) | +1 | |
| l_ank_pitch | (0, -1, 0) | -1 | |
| l_ank_roll | (+1, 0, 0) | +1 | |
| r_hip_yaw | (0, 0, -1) | -1 | |
| r_hip_roll | (-1, 0, 0) | -1 | |
| r_hip_pitch | (0, -1, 0) | -1 | |
| r_knee | (0, -1, 0) | -1 | |
| r_ank_pitch | (0, +1, 0) | +1 | |
| r_ank_roll | (+1, 0, 0) | +1 | |

## Critical Gotchas

### 1. Same Axis ≠ Same Visual Direction (Elbow Trap)

Both `l_el` and `r_el` have axis `X+`. But the arm geometry is **Y-mirrored** (left arm at Y=+0.06, right at Y=-0.06). This means:

- **Same positive angle** makes them bend in **visually opposite** directions
- `l_el = -0.8, r_el = +0.8` → both forearms bend **inward** (toward body) ✅
- `l_el = +0.8, r_el = +0.8` → one bends inward, one bends outward ❌
- `l_el = +0.8, r_el = -0.8` → both bend **outward** (away from body) ❌

**Rule**: When two mirrored joints share the same axis, test the actual physical result (forearm tip position), don't just check for "symmetry".

### 2. Arm Swing Counter-Phase (ROBOTIS Source)

ROBOTIS `computeArmAngle()` uses `getJointDirection()` to negate the right arm:

```cpp
// Right arm
arm_angle[0] = wSin(..., -x_move * gain * 1000, 0) * getJointDirection("r_sho_pitch"); // dir = -1
// Left arm  
arm_angle[1] = wSin(..., +x_move * gain * 1000, 0) * getJointDirection("l_sho_pitch"); // dir = +1
```

Since `r_sho_pitch` has dir=-1 and `l_sho_pitch` has dir=+1, both arms get the **same final sign**, which produces **opposite visual motion** (counter-phase) because the axes are mirrored Y+ vs Y-.

**Wrong approach**: Give both arms the same `arm_swing` value.
**Correct approach**: Negate one arm's swing: `r_sho_pitch = -arm_swing`.

### 3. Meta Virtual Monitor Breaks MuJoCo

If using a Meta Quest headset, the **Meta Virtual Monitor** display driver intercepts WGL (Windows OpenGL) and causes:
```
ERROR: could not create window
GLFWError: (65542) 'WGL: The driver does not appear to support OpenGL'
```

**Fix**: Disable "Meta Virtual Monitor" in Device Manager → Display adapters, then restart the NVIDIA GPU driver:
```
pnputil /restart-device "PCI\VEN_10DE&DEV_XXXX&..."
```

### 4. Position Symmetry ≠ Correct Direction

When testing joint directions, checking that positions are "symmetric" (same Z height) is **necessary but not sufficient**. Two configurations can both be symmetric but bend in opposite directions (one inward, one outward).

**Always check the actual direction** (e.g., does forearm tip Y move toward or away from body center?).

## Verification Script

Use `verify_all_joints.py` pattern:
1. Read all joint axes from MuJoCo model
2. Compare against ROBOTIS source definitions
3. Test each joint individually with simulation
4. Test combined arm poses with forearm tip tracking
5. Verify "inward" vs "outward" by comparing tip Y to elbow Y

Key test: for elbows with arms hanging at sides (`sho_roll` applied):
```python
# If forearm_tip_Y is closer to 0 than elbow_Y → inward bend ✅
# If forearm_tip_Y is farther from 0 than elbow_Y → outward bend ❌
```
