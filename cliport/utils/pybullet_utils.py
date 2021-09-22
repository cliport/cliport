"""PyBullet utilities for loading assets."""
import os
import six

import pybullet as p


# BEGIN GOOGLE-EXTERNAL
def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""
    # Handles most general file open case.
    try:
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass

# END GOOGLE-EXTERNAL
