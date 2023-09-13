import pickle
import cv2
import json
#import rospy

import numpy as np

from cliport.model_comparison.comparison_utilities import DataHandler, DataDrawer
#from cliport.ros_server import RigidTransformer, get_bbox
from transforms3d._gohlketransforms import quaternion_matrix, euler_from_quaternion, translation_matrix
from cliport.utils import utils_2
#from std_msgs.msg import String


ROOT_DIR = "/home/drubuntu/cliport"
RESULT_FOLDER = f"{ROOT_DIR}/comparison_results"
DATA_FOLDER = f"{ROOT_DIR}/data"
EXP_FOLDER = f"{ROOT_DIR}/exps"
CFG_FILE = "train.yaml"
MODELS = [
    "engine-parts-to-box-single-list",
    "engine-parts-single",
]
MODEL_EXTENDERS = [
    "-cliport-n88-train",
    "-cliport-n34-train",
]
AMALGAM_TITLE = "complete-unseen-amalgamation"

def get_bbox_allrad(point, offset, shape):
    """Get bbox given point and offset"""
    first = [point[0] - offset, point[1] - offset]
    second = [point[0] + offset, point[1] + offset]

    return [first, second]

def get_bbox_thresh(point, offset, shape):
    """Get bbox given point and offset"""

    return [
        [
            min(max(0, point[0] - offset), shape[1]), 
            min(max(0, point[1] - offset), shape[0])
        ], 
        [
            max(min(shape[1] - 1, point[0] + offset), 0), 
            max(min(shape[0] - 1, point[1] + offset), 0)
        ]
    ]

class RigidTransformer:

    def __init__(self):
        self.intrinsic_mat = np.array(
            [[609.9600830078125, 0.0, 336.7248229980469], [0.0, 609.9955444335938, 249.56271362304688],
             [0.0, 0.0, 1.0]])

        rotation_xyzw = [0.7163862506670556, -0.6969879890334542, -0.029152199866571128, 0.012191482323328342]
        #rotation_xyzw = [0.691011, -0.00811551, -0.029152199866571128, 0.012191482323328342]
        
        rotation_wxyz = [rotation_xyzw[-1], *rotation_xyzw[:-1]]

        translation_xyz = [-0.00172061, 0.34352981, 0.63233277]
        #translation_xyz = [0.0593305, -0.0202352, -0.135014]

        tmat, rmat = translation_matrix(translation_xyz), quaternion_matrix(rotation_wxyz)
        self.rigid_transform = np.dot(tmat, rmat)

    def xyz_to_pix(self, xyz):
        """Convert world coordinates to pixels"""
        intrinsic_mat = self.intrinsic_mat
        world_xyz = np.ones((4, 1))
        world_xyz[:3, 0] = xyz[:]
        camera_xyz = np.dot(self.rigid_transform, world_xyz).flatten()[:3] * 1000
        u = (intrinsic_mat[0][0] * camera_xyz[0]) / camera_xyz[2] + intrinsic_mat[0][2]
        v = (intrinsic_mat[1][1] * camera_xyz[1]) / camera_xyz[2] + intrinsic_mat[1][2]

        return int(round(u)), int(round(v))


def main() -> int:
    data_handler = DataHandler(        
        root=ROOT_DIR,
        exp_path=EXP_FOLDER,
        data_path=DATA_FOLDER,
        result_path=RESULT_FOLDER,
        cfg_filename=CFG_FILE
        )
    data_drawer = DataDrawer(cols=2, rows=2)    

    rt = RigidTransformer()

    #rospy.init_node("cliport", anonymous=True, log_level=rospy.INFO)
    #pose_pub = rospy.Publisher("/cliport/out", String, queue_size=3)

    for i in range(len(MODELS)):
        training_size = data_handler.get_set_limit(MODELS[i], "train")
        validation_size = data_handler.get_set_limit(AMALGAM_TITLE, "val")
        lang_goals = data_handler.get_lang_goals(MODELS[i])

        data_handler.read_dataset(MODELS[i], "train")
        data_handler.organize_set("train")
        data_handler.read_dataset(AMALGAM_TITLE, "val")
        data_handler.organize_set("val")
        data_handler.find_latest_best_checkpoint_version(MODELS[i], MODEL_EXTENDERS[i])
        data_handler.augment_cfg(MODELS[i], MODEL_EXTENDERS[i])
        data_handler.load_model(MODELS[i], MODEL_EXTENDERS[i])

        for episode_index in range(validation_size):
            episode = data_handler.get_observation(episode_index, "val")
            (obs, act_actual, _, info) = episode
            
            image = obs['color']
            depth = obs['depth']
            lang_goal = info['lang_goal']

            pick_real_xyz = act_actual['pose0'][0]
            place_real_xyz = act_actual['pose1'][0]
            pick_real_rotation = info['pick_data'][0]['rotation']
            place_real_rotation = info['place_data'][0]['rotation']
            
            act = data_handler.act_on_model(obs, info, goal=None)
            hmap = act['hmap']
            pick_confidence = act['pick_confidence']
            place_confidence = act['place_confidence']

            condition = True
            while condition:
                act = data_handler.act_on_model(obs, info, goal=None)
                pick_pred = rt.xyz_to_pix(act['pose0'][0])
                place_pred = rt.xyz_to_pix(act['pose1'][0])
                cfg = data_handler.agent.cam_config[0]
                intrinsics = np.array(cfg['intrinsics']).reshape((3, 3))
                extrinsics = {'xyz': cfg['position'], 'quaternion': cfg['rotation']}
                pick_pred_bbox = get_bbox_thresh(pick_pred, offset=32, shape=depth.shape)
                place_pred_bbox = get_bbox_thresh(place_pred, offset=32, shape=depth.shape)

                
                pick_pred_xyz, _ = utils_2.get_avg_3d_centroid(depth, pick_pred_bbox, intrinsics, extrinsics)
                place_pred_xyz, _ = utils_2.get_avg_3d_centroid(depth, place_pred_bbox, intrinsics, extrinsics)
                pick_pred_rotation = np.degrees(act['pick'][-1])
                place_pred_rotation = np.degrees(act['place'][-1])

                #pick_pred_xyz = [0 if a_ < 0 else a_ for a_ in pick_pred_xyz]
                #place_pred_xyz = [0 if a_ < 0 else a_ for a_ in place_pred_xyz]

                condition = np.isnan(pick_pred_xyz).any() or np.isnan(place_pred_xyz).any()

                #rospy.loginfo(f"pose0:{pick_pred_xyz}")
                #rospy.loginfo(f"pose1:{place_pred_xyz}")
                #if condition:
                #    rospy.loginfo("ACTED AGAIN, BAD LOCATION")

            data_dic = {
                'pick_xyz': pick_pred_xyz, 
                'pick_rotation': pick_pred_rotation, 
                'place_xyz': place_pred_xyz,
                'place_rotation': place_pred_rotation
                }
            
            #data_str = json.dumps(data_dic)
            #pose_pub.publish(data_str)

            data_drawer.set_axs(0, 0)
            data_drawer.draw_data_to_active_axs(image)
            data_drawer.set_axs(0, 1)
            data_drawer.draw_data_to_active_axs(hmap)
            data_drawer.set_axs(1, 0)
            data_drawer.draw_data_to_active_axs(pick_confidence)
            data_drawer.set_axs(1, 1)
            data_drawer.draw_data_to_active_axs(place_confidence)

            input("Continue?")

    return 0




"""
with open('/home/opendr/mikael/cliport/data/complete-unseen-amalgamation-val/0001.pkl', 'rb') as f:
    datademo = pickle.load(f)


image = datademo['color']
depth = datademo['depth']
lang_goal = datademo['info']['lang_goal']
place_data = datademo['info']['place_data']
pick_data = datademo['info']['pick_data']

imv = cv2.rectangle(image, pick_data['bbox'][0], pick_data['bbox'][1], (255, 0, 0), 2)
imv = cv2.rectangle(imv, place_data['bbox'][0], place_data['bbox'][1], (0, 255, 0), 2)


cv2.imshow('image', imv)
cv2.waitKey(0)
"""
if __name__ == '__main__':
    main()
