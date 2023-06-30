"""Camera handling"""
import time
import pickle
import cv2
import numpy as np
from pathlib import Path

# Cliport imports
from cliport.utils import utils_2

from transforms3d._gohlketransforms import quaternion_matrix, euler_from_quaternion, translation_matrix


class RigidTransformer:

    def __init__(self):
        self.intrinsic_mat = np.array([[609.9600830078125, 0.0, 336.7248229980469], [0.0, 609.9955444335938, 249.56271362304688], [0.0, 0.0, 1.0]])
        

        rotation_xyzw = [0.7163862506670556, -0.6969879890334542, -0.029152199866571128, 0.012191482323328342]

        rotation_wxyz = [rotation_xyzw[-1], *rotation_xyzw[:-1]]

        translation_xyz = [-0.00172061,  0.34352981,  0.63233277]

        tmat, rmat = translation_matrix(translation_xyz), quaternion_matrix(rotation_wxyz)
        self.rigid_transform = np.dot(tmat, rmat)

    def xyz_to_pix(self, xyz):
        """Convert world coordinates to pixels"""
        intrinsic_mat = self.intrinsic_mat
        world_xyz = np.ones((4, 1))
        world_xyz[:3, 0] = xyz[:]
        camera_xyz = np.dot(self.rigid_transform, world_xyz).flatten()[:3]*1000
        u = (intrinsic_mat[0][0]*camera_xyz[0])/camera_xyz[2] + intrinsic_mat[0][2]
        v = (intrinsic_mat[1][1]*camera_xyz[1])/camera_xyz[2] + intrinsic_mat[1][2]

        return int(round(u)), int(round(v))



def read_pickle_file(filename):
    data = None
    with open(filename, 'rb') as fd:
        data = pickle.load(fd)
    return data

def main() -> None:
    """Main program flow"""
    # Initialize classes
    # Main program loop
    rt = RigidTransformer()
    files = list(Path('results/').rglob('*.pkl'))
    for file in files:
        data = read_pickle_file(file)
        if data is None:
            continue
        rgb = data['obs']['color']
        lang_goal = data['obs']['lang_goal']
        act = data['act']
        runtime = data['runtime']

        pick = rt.xyz_to_pix(act['pose0'][0]) # width,height order of pixels
        place = rt.xyz_to_pix(act['pose1'][0])

        # Draw and View Image
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.circle(bgr, pick, 5, (0, 255, 0), -1)
        bgr = cv2.circle(bgr, place, 5, (255, 0, 0), -1)

        cv2.imshow('Image', bgr)
        print('Runtime: ', runtime)
        print('Language Goal: ', lang_goal)
        print('Output Actions: ', act)

        key_press = cv2.waitKey(0) & 0xFF
        if key_press == ord('q'):
            break


if __name__ == '__main__':
    main()