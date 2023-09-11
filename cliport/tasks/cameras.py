"""Camera configs."""

import numpy as np
import pybullet as p


class RealSenseD415():
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Set default camera poses.
    front_position = (1., 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)

    # Default camera configs.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': front_position,
        'rotation': front_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': left_position,
        'rotation': left_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': right_position,
        'rotation': right_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]


class Oracle():
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
    position = (0.5, 0, 1000.)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (999.7, 1001.),
        'noise': False
    }]


class RS200Gazebo():
    """Gazebo Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (554.3826904296875, 0.0, 320.0, 0.0, 554.3826904296875, 240.0, 0.0, 0.0, 1.0)
    position = (0.5, 0, 1.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, np.pi / 2))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]


class KinectFranka():
    """Kinect Franka Camera"""

    # Near-orthographic projection.
    image_size = (424,512)
    intrinsics = (365.57489013671875, 0.0, 257.5205078125, 0.0, 365.57489013671875, 205.26710510253906, 0.0, 0.0, 1.0)
    position = (1.082, -0.041, 1.027)
    rotation = p.getQuaternionFromEuler((-2.611, 0.010, 1.553))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]


class D435Franka():
    """D435 Franka Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (609.9600830078125, 0.0, 336.7248229980469, 0.0, 609.9955444335938, 249.56271362304688, 0.0, 0.0, 1.0)
    position = (0.0593305, -0.0202352, -0.135014)
    rotation = (0.691011, -0.00811551, 0.722766, 0.006905)

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (0.1, 7.0),
        'noise': False
    }]

class D415Franka():
    """D415 Franka Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (609.9600830078125, 0.0, 336.7248229980469, 0.0, 609.9955444335938, 249.56271362304688, 0.0, 0.0, 1.0)
    position = (0.3588953639276142, -0.028871636526010304, 0.6240266185292889)
    rotation = (-0.7156933247588257, 0.6977166824649315, 0.028417185903828204, 0.012917065692708164)

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]