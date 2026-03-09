



"""
OpenServoSim - Base Controller Interface

All controllers must inherit from BaseController and implement
the compute() method. This ensures a consistent API across
different control algorithms (UVC, CPG, RL, etc.).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """
    Abstract base class for servo robot controllers.

    A controller takes sensor data (IMU, joint states) and produces
    joint position targets for the servo actuators.
    """

    def __init__(self, num_joints: int = 10, control_freq: float = 50.0):
        """
        Args:
            num_joints: Number of controllable joints
            control_freq: Control loop frequency in Hz
        """
        self.num_joints = num_joints
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

    @abstractmethod
    def compute(
        self,
        imu_data: dict,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> np.ndarray:
        """
        Compute joint position targets.

        Args:
            imu_data: dict with 'pitch', 'roll', 'yaw' (radians), 'gyro' (rad/s)
            joint_positions: Current joint angles in radians, shape (num_joints,)
            joint_velocities: Current joint velocities in rad/s, shape (num_joints,)

        Returns:
            Target joint positions in radians, shape (num_joints,)
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset controller internal state. Override in subclasses."""
        pass

    def get_info(self) -> dict:
        """
        Return debug information about the controller state.
        Override in subclasses to expose internal variables.
        """
        return {}
