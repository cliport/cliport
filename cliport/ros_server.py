"""Camera handling"""
from enum import Enum
import os
import time
import pickle
import json
from threading import Lock
# Ros libraries
import rospy
from cv_bridge import CvBridge

# Ros Messages
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import numpy as np
import torch

# Cliport imports
from cliport import agents
from cliport.utils import utils
from cliport.utils import utils_2

from transforms3d._gohlketransforms import quaternion_matrix, euler_from_quaternion, translation_matrix


class RigidTransformer:

    def __init__(self):
        self.intrinsic_mat = np.array(
            [[609.9600830078125, 0.0, 336.7248229980469], [0.0, 609.9955444335938, 249.56271362304688],
             [0.0, 0.0, 1.0]])

        rotation_xyzw = [0.7163862506670556, -0.6969879890334542, -0.029152199866571128, 0.012191482323328342]

        rotation_wxyz = [rotation_xyzw[-1], *rotation_xyzw[:-1]]

        translation_xyz = [-0.00172061, 0.34352981, 0.63233277]

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


class CameraStream:

    def __init__(self):
        """Initialize ros subscriber"""

        # subscribed Topic
        topic_rgb = "/camera/color/image_raw"
        topic_depth = "/camera/aligned_depth_to_color/image_raw"
        self.subscriber_rgb = rospy.Subscriber(topic_rgb,
                                               Image, self.callback_rgb, queue_size=1, buff_size=2 ** 32)
        self.subscriber_depth = rospy.Subscriber(topic_depth,
                                                 Image, self.callback_depth, queue_size=1, buff_size=2 ** 32)

        self.bridge = CvBridge()
        self.rgb = self.depth = self.hmap = self.pick_confidence = self.place_confidence = None
        rospy.loginfo(f"Subscribed to {topic_rgb}")
        rospy.loginfo(f"Subscribed to {topic_depth}")

    def callback_rgb(self, data):
        """RGB callback"""
        self.rgb = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")

    def callback_depth(self, data):
        """Depth callback"""
        self.depth = self.bridge.imgmsg_to_cv2(data, )

    def set_hmap(self, hmap):
        self.hmap = hmap

    def set_confidences(self, pick_confidence, place_confidence):
        self.pick_confidence = pick_confidence
        self.place_confidence = place_confidence


class StreamType(Enum):
    RGB = 0
    DEPTH = 1
    HMAP = 2


class ToolGUI:
    """Visualize and interact with camera stream and snapshot window"""

    def __init__(self, streamer) -> None:
        """Initialize stuff"""
        self.stream_win = "RGB stream"
        self.hmap_win = "Heightmap capture"
        self.pick_conf_win = "pick task confidence"
        self.place_conf_win = "place task confidence"
        self.streamer = streamer
        cv2.namedWindow(self.stream_win)

    def cleanup(self):
        """Release resources"""
        cv2.destroyAllWindows()

    def run(self, pick, place) -> None:
        """Run GUI"""
        self.handle_stream(pick, place)
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            raise KeyboardInterrupt

    def handle_stream(self, pick, place) -> None:
        """Display and interact with stream window"""
        if self.streamer.rgb is not None:
            # OpenCV handles bgr format instead of rgb, so we convert first
            bgr = cv2.cvtColor(self.streamer.rgb, cv2.COLOR_RGB2BGR)
            if pick is not None and place is not None:
                bgr = cv2.circle(bgr, pick, 5, (0, 255, 0), -1)
                bgr = cv2.circle(bgr, place, 5, (255, 0, 0), -1)
                # print('Pixel Coordinates: ', pick, place)
            cv2.namedWindow(self.stream_win, cv2.WINDOW_KEEPRATIO)
            cv2.imshow(self.stream_win, bgr)

        if self.streamer.hmap is not None:
            hmap = self.streamer.hmap
            hmap_norm = hmap / np.max(hmap)
            hmap_norm = np.matrix.transpose(np.flip(hmap_norm))
            cv2.imshow(self.hmap_win, hmap_norm)

        if self.streamer.pick_confidence is not None:
            pick_confidence = self.streamer.pick_confidence
            pick_reshape = np.matrix.transpose(np.flip(pick_confidence))[0]
            pick_aug = pick_reshape / pick_reshape.max()            # normalize values so that max is 1
            #pick_aug = pick_reshape / np.average(pick_reshape)
            #pick_aug = np.clip(pick_aug, 0, 1)
            im = np.array(pick_aug * 255, dtype=np.uint8)           # transform into 0-255 grayscale
            im = cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)     # normalize with cv2 for good measure
            im = cv2.applyColorMap(im, cv2.COLORMAP_PLASMA)         # colormap
            cv2.imshow(self.pick_conf_win, im)

        if self.streamer.place_confidence is not None:
            place_confidence = self.streamer.place_confidence
            place_reshape = np.matrix.transpose(np.flip(place_confidence))[0]
            place_aug = place_reshape / place_reshape.max()
            #place_aug = place_reshape / np.average(place_reshape)
            #place_aug = np.clip(place_aug, 0, 1)
            im = np.array(place_aug * 255, dtype=np.uint8)
            im = cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
            im = cv2.applyColorMap(im, cv2.COLORMAP_PLASMA)
            cv2.imshow(self.place_conf_win, im)

        # if self.streamer.depth is not None and self.stream_type is StreamType.DEPTH:
        #    cv2.imshow(self.stream_win, depth_to_heatmap(self.streamer.depth))


class CLIPORT:
    """CLIPORT agent class"""

    def __init__(self):
        """Init agent"""
        agent_type = 'two_stream_clip_lingunet_lat_transporter'
        model_folder = 'exps/engine-parts-to-box-single-list-cliport-n88-train/checkpoints'
        ckpt_name = 'best.ckpt'  # name of checkpoint to load
        eval_task = 'packing-objects'
        root_dir = os.environ['CLIPORT_ROOT']
        config_file = 'train.yaml'
        cfg = utils.load_hydra_config(os.path.join(root_dir, f'cliport/cfg/{config_file}'))
        name = '{}-{}'.format(eval_task, agent_type, )
        agent = agents.names[agent_type](name, cfg, None, None)
        # Load checkpoint
        ckpt_path = os.path.join(root_dir, model_folder, ckpt_name)
        rospy.loginfo(f'\nLoading checkpoint: {ckpt_path}')
        agent.load(ckpt_path)

        self.agent = agent
        self.pick = None
        self.place = None
        self.act = None
        self.hmap = None
        self.pick_confidence = None
        self.place_confidence = None

        # sub/pub
        self.pose_pub = rospy.Publisher("/cliport/out", String, queue_size=3)
        self.lang_sub = rospy.Subscriber("/cliport/in", String, self.lang_callback)
        self.lang = None

    def lang_callback(self, msg):
        """Get language input from cliport client"""
        rospy.loginfo(f"Got input msg: {msg.data}")
        if msg.data == 'pick a white box':
            self.lang = 'put white box in brown box'
        else:
            self.lang = msg.data

    def run(self, obs):
        """Run model inference"""
        if obs['color'] is None or obs['depth'] is None:
            rospy.logwarn("CLIPORT inference failed. Invalid rgb-depth observation supplied.")
        if self.lang is None:
            rospy.logwarn("CLIPORT inference failed. language input is Empty")

        act = self.agent.act(obs, {'lang_goal': self.lang}, goal=None)
        self.act = act
        rospy.loginfo("COUTACT")
        rospy.loginfo(act)
        self.pick = act['pose0']
        self.place = act['pose1']
        self.hmap = act['hmap']
        self.pick_confidence = act['pick_confidence']
        self.place_confidence = act['place_confidence']


def get_bbox(point, offset, shape):
    """Get bbox given point and offset"""
    bbox = []
    first = [max(0, point[0] - offset), max(0, point[1] - offset)]
    second = [min(shape[1] - 1, point[0] + offset), min(shape[0] - 1, point[1] + offset)]

    return [first, second]


# Orientation of end effector at default pose. HARDCODED! Should be published from master node
# REF_ORIENTATION_WXYZ = [0.018651850446173072, -0.997114679759071, 0.07321044029759569, -0.007392923327828006]


def write_pickle_file(data):
    filename = time.strftime("%Y%m%d-%H%M%S") + ".pkl"
    with open(f'/home/opendr/omar/Work/cliport/results/{filename}', 'wb') as fd:
        pickle.dump(data, fd)


def main() -> None:
    """Main program flow"""
    write_image = False
    # Create ros node
    rospy.init_node("cliport", anonymous=True, log_level=rospy.INFO)
    # Initialize classes
    # Main program loop
    rt = RigidTransformer()
    cliport = CLIPORT()
    streamer = CameraStream()
    gui = ToolGUI(streamer)
    try:
        pick = place = None
        cliport.lang = 'put bolt in brown box'
        while not rospy.is_shutdown():
            if streamer.rgb is not None and streamer.depth is not None and cliport.lang is not None:
                t1 = time.time()
                rgb = streamer.rgb.copy()
                depth = streamer.depth.copy()
                obs = {'color': rgb, 'depth': depth}
                cliport.run(obs)
                picks = []
                places = []
                key_list = list(cliport.act.keys())
                pose_key_list = []
                """
                # extract 'poseX' keys from dict with other entries
                for entry in key_list:
                    if "pose" in entry:
                        pose_key_list.append(entry)
                # add every pick (poseX) & place (poseX+1) pair from dict to relevant lists
                for i in range(len(pose_key_list)):
                    if i % 2 == 0:
                        picks.append(rt.xyz_to_pix(cliport.act[f'pose{i}'][0])) # width,height order of pixels
                    else:
                        places.append(rt.xyz_to_pix(cliport.act[f'pose{i}'][0]))
                """
                pick = rt.xyz_to_pix(cliport.act['pose0'][0])  # width,height order of pixels
                place = rt.xyz_to_pix(cliport.act['pose1'][0])
                streamer.set_hmap(cliport.act['hmap'])
                streamer.set_confidences(cliport.act['pick_confidence'], cliport.act['place_confidence'])

                cfg = cliport.agent.cam_config[0]
                intrinsics = np.array(cfg['intrinsics']).reshape((3, 3))
                extrinsics = {'xyz': cfg['position'], 'quaternion': cfg['rotation']}

                # Get 3d centroid for each pick/place
                # for i in range(len(places)):

                # pick = picks[i]
                # place = places[i]
                pick_bbox = get_bbox(pick, offset=32, shape=depth.shape)
                place_bbox = get_bbox(place, offset=32, shape=depth.shape)
                pick_xyz, _ = utils_2.get_avg_3d_centroid(depth, pick_bbox, intrinsics, extrinsics)
                place_xyz, _ = utils_2.get_avg_3d_centroid(depth, place_bbox, intrinsics, extrinsics)

                pick_rotation = np.degrees(cliport.act['pick'][-1])
                place_rotation = np.degrees(cliport.act['place'][-1])

                data_dic = {'pick_xyz': pick_xyz, 'pick_rotation': pick_rotation, 'place_xyz': place_xyz,
                            'place_rotation': place_rotation}

                t2 = time.time()

                data_str = json.dumps(data_dic)

                print(data_str)
                cliport.pose_pub.publish(data_str)

                rospy.loginfo(f"Published {data_str} to CLIPORT client")
                rospy.loginfo(f"Runtime: {t2 - t1}")
                rospy.loginfo(f"Writing result data to file...")
                writable_data = {'obs': {'color': rgb, 'depth': depth, 'lang_goal': cliport.lang},
                                 'act': cliport.act, 'runtime': t2 - t1}
                write_pickle_file(writable_data)

                cliport.lang = None
            gui.run(pick, place)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down cliport server")
    gui.cleanup()


if __name__ == '__main__':
    os.environ["ROS_MASTER_URI"] = "http://172.16.0.10:11311"
    # os.environ["ROS_MASTER_URI"] = "172.16.0.10:42711"
    os.environ["ROS_IP"] = "172.16.0.11"

    main()