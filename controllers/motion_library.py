"""
Guge Upper Body Robot - Motion Library (v2)

Keyframe-based motion system with cubic spline interpolation.

JOINT DIRECTION REFERENCE (verified at +45°):
  spine1:     (+) = rotate left (viewer perspective)
  spine2:     (+) = bow forward
  spine3:     (+) = lateral tilt

  left_arm1:  (+) = raise arm (swing up/outward)
  left_arm2:  (+) = forward flexion + raise
  left_arm3:  (+) = upper arm internal rotation
  left_arm4:  (+) = bend elbow inward
  left_arm5:  (+) = wrist rotation (away from body)

  right_arm1: (+) = LOWER arm (swing down) ← NOTE: opposite of left!
  right_arm2: (+) = complex rotation
  right_arm3: (+) = upper arm rotation
  right_arm4: (+) = bend elbow inward
  right_arm5: (+) = wrist rotation

Camera: azimuth=90 is FRONT view.
Robot's left arm = viewer's RIGHT. Robot's right arm = viewer's LEFT.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import math

JOINT_NAMES = [
    "spine1", "spine2", "spine3",
    "left_arm1", "left_arm2", "left_arm3", "left_arm4", "left_arm5",
    "right_arm1", "right_arm2", "right_arm3", "right_arm4", "right_arm5",
]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
NUM_JOINTS = len(JOINT_NAMES)

def deg(d):
    return math.radians(d)


@dataclass
class Keyframe:
    time: float
    joints: Dict[str, float]


@dataclass
class Motion:
    name: str
    description: str
    keyframes: List[Keyframe]
    loop: bool = False


def _cubic_hermite(t, p0, p1, m0, m1):
    t2 = t * t
    t3 = t2 * t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    return h00*p0 + h10*m0 + h01*p1 + h11*m1


class MotionLibrary:
    def __init__(self):
        self._motions: Dict[str, Motion] = {}
        self._register_all_motions()

    def get(self, name: str) -> Motion:
        if name not in self._motions:
            raise KeyError(f"Unknown motion: {name}. Available: {list(self._motions.keys())}")
        return self._motions[name]

    def list_motions(self) -> List[str]:
        return list(self._motions.keys())

    def get_duration(self, motion: Motion) -> float:
        return motion.keyframes[-1].time

    def evaluate(self, motion: Motion, t: float) -> np.ndarray:
        kfs = motion.keyframes
        duration = kfs[-1].time
        if motion.loop and duration > 0:
            t = t % duration
        t = np.clip(t, 0, duration)

        idx = 0
        for i in range(len(kfs) - 1):
            if kfs[i + 1].time >= t:
                idx = i
                break
        else:
            idx = len(kfs) - 2

        kf0, kf1 = kfs[idx], kfs[idx + 1]
        dt = kf1.time - kf0.time
        if dt < 1e-6:
            return self._keyframe_to_array(kf1)

        s = (t - kf0.time) / dt
        v0 = self._keyframe_to_array(kf0)
        v1 = self._keyframe_to_array(kf1)

        # Catmull-Rom tangents
        if idx > 0:
            v_prev = self._keyframe_to_array(kfs[idx - 1])
            dt_prev = kf0.time - kfs[idx - 1].time
            m0 = dt * (v1 - v_prev) / (dt + dt_prev + 1e-6)
        else:
            m0 = np.zeros(NUM_JOINTS)

        if idx + 2 < len(kfs):
            v_next = self._keyframe_to_array(kfs[idx + 2])
            dt_next = kfs[idx + 2].time - kf1.time
            m1 = dt * (v_next - v0) / (dt + dt_next + 1e-6)
        else:
            m1 = np.zeros(NUM_JOINTS)

        result = np.zeros(NUM_JOINTS)
        for j in range(NUM_JOINTS):
            result[j] = _cubic_hermite(s, v0[j], v1[j], m0[j], m1[j])
        return result

    def _keyframe_to_array(self, kf: Keyframe) -> np.ndarray:
        arr = np.zeros(NUM_JOINTS)
        for name, angle in kf.joints.items():
            if name in JOINT_INDEX:
                arr[JOINT_INDEX[name]] = angle
        return arr

    def _register(self, motion: Motion):
        self._motions[motion.name] = motion

    # ================================================================
    #  ALL KEYFRAMES USE SMALL SPINE ANGLES (< 15°)
    #  Arms use verified directions
    # ================================================================

    def _register_all_motions(self):
        self._register(self._motion_wave_hello())
        self._register(self._motion_bow())
        self._register(self._motion_welcome())
        self._register(self._motion_nod())
        self._register(self._motion_shake_head())
        self._register(self._motion_surprise())
        self._register(self._motion_thinking())
        self._register(self._motion_clap())
        self._register(self._motion_point_left())
        self._register(self._motion_point_right())
        self._register(self._motion_wave_goodbye())
        self._register(self._motion_bye_bye())

    # --- 1. 挥手问好 (Wave Hello) ---
    # Right arm (viewer's left) raises and waves
    # right_arm1(-) = raise, right_arm4(-) = bend elbow outward
    def _motion_wave_hello(self) -> Motion:
        return Motion(
            name="wave_hello",
            description="右臂抬起挥手问好",
            keyframes=[
                Keyframe(0.0, {}),
                # Raise right arm
                Keyframe(0.8, {
                    "right_arm1": deg(-70),   # raise (negative = up for right arm)
                    "right_arm2": deg(20),
                    "right_arm4": deg(-50),   # bend elbow
                }),
                # Wave 1
                Keyframe(1.2, {
                    "right_arm1": deg(-70),
                    "right_arm2": deg(20),
                    "right_arm4": deg(-50),
                    "right_arm5": deg(35),
                }),
                # Wave 2
                Keyframe(1.6, {
                    "right_arm1": deg(-70),
                    "right_arm2": deg(20),
                    "right_arm4": deg(-50),
                    "right_arm5": deg(-35),
                }),
                # Wave 3
                Keyframe(2.0, {
                    "right_arm1": deg(-70),
                    "right_arm2": deg(20),
                    "right_arm4": deg(-50),
                    "right_arm5": deg(35),
                }),
                # Wave 4
                Keyframe(2.4, {
                    "right_arm1": deg(-70),
                    "right_arm2": deg(20),
                    "right_arm4": deg(-50),
                    "right_arm5": deg(-35),
                }),
                # Return
                Keyframe(3.2, {}),
            ]
        )

    # --- 2. 鞠躬 (Bow) ---
    # spine2(+) = bow forward, keep small angle
    def _motion_bow(self) -> Motion:
        return Motion(
            name="bow",
            description="腰部前倾鞠躬",
            keyframes=[
                Keyframe(0.0, {}),
                Keyframe(0.8, {"spine2": deg(20)}),   # gentle bow
                Keyframe(2.0, {"spine2": deg(20)}),    # hold
                Keyframe(3.0, {}),                      # return
            ]
        )

    # --- 3. 双手欢迎 (Welcome) ---
    # Both arms raise: left_arm1(+)=raise, right_arm1(-)=raise
    def _motion_welcome(self) -> Motion:
        return Motion(
            name="welcome",
            description="双臂展开欢迎",
            keyframes=[
                Keyframe(0.0, {}),
                # Spread arms wide
                Keyframe(0.8, {
                    "left_arm1": deg(50),     # raise left arm (viewer right)
                    "left_arm2": deg(30),
                    "right_arm1": deg(-50),   # raise right arm (viewer left)
                    "right_arm2": deg(30),
                }),
                # Hold
                Keyframe(1.6, {
                    "left_arm1": deg(50),
                    "left_arm2": deg(30),
                    "right_arm1": deg(-50),
                    "right_arm2": deg(30),
                }),
                # Arms forward (inviting gesture)
                Keyframe(2.4, {
                    "left_arm1": deg(20),
                    "left_arm2": deg(50),
                    "left_arm4": deg(20),
                    "right_arm1": deg(-20),
                    "right_arm2": deg(50),
                    "right_arm4": deg(-20),
                }),
                # Slight bow with arms
                Keyframe(3.0, {
                    "spine2": deg(8),
                    "left_arm1": deg(20),
                    "left_arm2": deg(50),
                    "left_arm4": deg(20),
                    "right_arm1": deg(-20),
                    "right_arm2": deg(50),
                    "right_arm4": deg(-20),
                }),
                # Return
                Keyframe(4.0, {}),
            ]
        )

    # --- 4. 点头 (Nod) ---
    # spine2(+) = forward, spine2(-) = back. Small angles.
    def _motion_nod(self) -> Motion:
        return Motion(
            name="nod",
            description="连续点头",
            keyframes=[
                Keyframe(0.0, {}),
                Keyframe(0.3, {"spine2": deg(10)}),
                Keyframe(0.5, {"spine2": deg(-3)}),
                Keyframe(0.8, {"spine2": deg(10)}),
                Keyframe(1.0, {"spine2": deg(-3)}),
                Keyframe(1.3, {"spine2": deg(10)}),
                Keyframe(1.5, {"spine2": deg(-3)}),
                Keyframe(2.0, {}),
            ]
        )

    # --- 5. 摇头 (Shake Head) ---
    # spine1(+) = rotate left, spine1(-) = rotate right. Small angles.
    def _motion_shake_head(self) -> Motion:
        return Motion(
            name="shake_head",
            description="左右摇头",
            keyframes=[
                Keyframe(0.0, {}),
                Keyframe(0.3, {"spine1": deg(12)}),
                Keyframe(0.6, {"spine1": deg(-12)}),
                Keyframe(0.9, {"spine1": deg(12)}),
                Keyframe(1.2, {"spine1": deg(-12)}),
                Keyframe(1.5, {"spine1": deg(10)}),
                Keyframe(1.8, {"spine1": deg(-10)}),
                Keyframe(2.2, {}),
            ]
        )

    # --- 6. 惊讶 (Surprise) ---
    # Both arms raise quickly
    def _motion_surprise(self) -> Motion:
        return Motion(
            name="surprise",
            description="双臂快速上举表示惊讶",
            keyframes=[
                Keyframe(0.0, {}),
                # Quick raise
                Keyframe(0.4, {
                    "left_arm1": deg(80),
                    "left_arm2": deg(15),
                    "right_arm1": deg(-80),
                    "right_arm2": deg(15),
                    "spine2": deg(-3),    # lean back slightly
                }),
                # Hold surprise
                Keyframe(1.3, {
                    "left_arm1": deg(80),
                    "left_arm2": deg(15),
                    "right_arm1": deg(-80),
                    "right_arm2": deg(15),
                    "spine2": deg(-3),
                }),
                # Slowly lower
                Keyframe(2.5, {}),
            ]
        )

    # --- 7. 思考 (Thinking) ---
    # Right arm to chin: right_arm1(-)=raise, right_arm4(-) = bend elbow
    def _motion_thinking(self) -> Motion:
        return Motion(
            name="thinking",
            description="右手托腮思考",
            keyframes=[
                Keyframe(0.0, {}),
                # Right arm to chin position
                Keyframe(0.8, {
                    "right_arm1": deg(-30),
                    "right_arm2": deg(50),
                    "right_arm3": deg(-20),
                    "right_arm4": deg(-70),
                    "spine1": deg(-8),    # slight turn right
                    "spine3": deg(5),     # slight tilt
                }),
                # Hold thinking
                Keyframe(2.5, {
                    "right_arm1": deg(-30),
                    "right_arm2": deg(50),
                    "right_arm3": deg(-20),
                    "right_arm4": deg(-70),
                    "spine1": deg(-8),
                    "spine3": deg(5),
                }),
                # Slight nod (eureka!)
                Keyframe(3.0, {
                    "right_arm1": deg(-30),
                    "right_arm2": deg(50),
                    "right_arm3": deg(-20),
                    "right_arm4": deg(-70),
                    "spine2": deg(8),
                }),
                # Return
                Keyframe(4.0, {}),
            ]
        )

    # --- 8. 鼓掌 (Clap) ---
    # Arms forward, then open/close. left_arm2(+)=forward, right_arm2(+)=...
    def _motion_clap(self) -> Motion:
        kfs = [Keyframe(0.0, {})]
        # Prepare - arms forward
        kfs.append(Keyframe(0.5, {
            "left_arm2": deg(50),
            "left_arm1": deg(20),
            "right_arm2": deg(50),
            "right_arm1": deg(-20),
        }))
        # 4 claps (reduced from 5 for cleaner motion)
        for i in range(4):
            t_base = 0.7 + i * 0.4
            # Open
            kfs.append(Keyframe(t_base, {
                "left_arm2": deg(50),
                "left_arm1": deg(40),
                "right_arm2": deg(50),
                "right_arm1": deg(-40),
            }))
            # Close (clap)
            kfs.append(Keyframe(t_base + 0.18, {
                "left_arm2": deg(50),
                "left_arm1": deg(5),
                "right_arm2": deg(50),
                "right_arm1": deg(-5),
            }))
        # Return
        kfs.append(Keyframe(kfs[-1].time + 0.6, {}))
        return Motion(name="clap", description="双手前伸鼓掌", keyframes=kfs)

    # --- 9. 左手指引 (Point Left) ---
    # Robot's left arm = viewer's right. left_arm1(+) = raise
    def _motion_point_left(self) -> Motion:
        return Motion(
            name="point_left",
            description="左臂伸出指向左方",
            keyframes=[
                Keyframe(0.0, {}),
                Keyframe(0.5, {"spine1": deg(10)}),  # turn slightly left
                Keyframe(1.0, {
                    "spine1": deg(10),
                    "left_arm1": deg(30),
                    "left_arm2": deg(40),
                }),
                # Hold
                Keyframe(2.2, {
                    "spine1": deg(10),
                    "left_arm1": deg(30),
                    "left_arm2": deg(40),
                }),
                Keyframe(3.0, {}),
            ]
        )

    # --- 10. 右手指引 (Point Right) ---
    # Robot's right arm = viewer's left. right_arm1(-) = raise
    def _motion_point_right(self) -> Motion:
        return Motion(
            name="point_right",
            description="右臂伸出指向右方",
            keyframes=[
                Keyframe(0.0, {}),
                Keyframe(0.5, {"spine1": deg(-10)}),  # turn slightly right
                Keyframe(1.0, {
                    "spine1": deg(-10),
                    "right_arm1": deg(-30),
                    "right_arm2": deg(40),
                }),
                # Hold
                Keyframe(2.2, {
                    "spine1": deg(-10),
                    "right_arm1": deg(-30),
                    "right_arm2": deg(40),
                }),
                Keyframe(3.0, {}),
            ]
        )

    # --- 11. 挥手再见 (Wave Goodbye) ---
    # Right arm high wave with slow body sway
    def _motion_wave_goodbye(self) -> Motion:
        return Motion(
            name="wave_goodbye",
            description="右臂高举慢速挥手再见",
            keyframes=[
                Keyframe(0.0, {}),
                # Raise arm high
                Keyframe(0.7, {
                    "right_arm1": deg(-80),
                    "right_arm2": deg(15),
                }),
                # Slow wave 1
                Keyframe(1.3, {
                    "right_arm1": deg(-80),
                    "right_arm2": deg(15),
                    "right_arm5": deg(40),
                    "spine1": deg(5),
                }),
                # Slow wave 2
                Keyframe(1.9, {
                    "right_arm1": deg(-80),
                    "right_arm2": deg(15),
                    "right_arm5": deg(-40),
                    "spine1": deg(-5),
                }),
                # Slow wave 3
                Keyframe(2.5, {
                    "right_arm1": deg(-80),
                    "right_arm2": deg(15),
                    "right_arm5": deg(40),
                    "spine1": deg(5),
                }),
                # Lower + slight bow
                Keyframe(3.5, {"spine2": deg(8)}),
                Keyframe(4.0, {}),
            ]
        )

    # --- 12. 双手拜拜 (Bye Bye) ---
    # Both arms raise and alternate wave
    def _motion_bye_bye(self) -> Motion:
        kfs = [Keyframe(0.0, {})]
        # Raise both arms
        kfs.append(Keyframe(0.5, {
            "left_arm1": deg(60),
            "left_arm2": deg(20),
            "left_arm4": deg(35),
            "right_arm1": deg(-60),
            "right_arm2": deg(20),
            "right_arm4": deg(-35),
        }))
        # Alternating waves
        for i in range(4):
            t_base = 0.7 + i * 0.45
            kfs.append(Keyframe(t_base, {
                "left_arm1": deg(60),
                "left_arm2": deg(20),
                "left_arm4": deg(35),
                "left_arm5": deg(30),
                "right_arm1": deg(-60),
                "right_arm2": deg(20),
                "right_arm4": deg(-35),
                "right_arm5": deg(-30),
            }))
            kfs.append(Keyframe(t_base + 0.22, {
                "left_arm1": deg(60),
                "left_arm2": deg(20),
                "left_arm4": deg(35),
                "left_arm5": deg(-30),
                "right_arm1": deg(-60),
                "right_arm2": deg(20),
                "right_arm4": deg(-35),
                "right_arm5": deg(30),
            }))
        kfs.append(Keyframe(kfs[-1].time + 0.7, {}))
        return Motion(name="bye_bye", description="双手举起交替挥动拜拜", keyframes=kfs)


if __name__ == "__main__":
    lib = MotionLibrary()
    print(f"Available motions ({len(lib.list_motions())}):")
    for name in lib.list_motions():
        m = lib.get(name)
        dur = lib.get_duration(m)
        n_kf = len(m.keyframes)
        print(f"  {name:<20s}  {dur:5.1f}s  {n_kf:2d} kf  - {m.description}")

    motion = lib.get("wave_hello")
    for t in np.arange(0, 3.2, 0.5):
        angles = lib.evaluate(motion, t)
        non_zero = np.count_nonzero(np.abs(angles) > 0.01)
        print(f"  t={t:.1f}s  active={non_zero}  max={np.degrees(np.max(np.abs(angles))):.1f}°")
