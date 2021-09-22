"""Sweeping Piles task."""

import numpy as np
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p


class SeparatingPilesUnseenColors(Task):
    """Separating Piles task."""

    def __init__(self):
        super().__init__()
        self.ee = Spatula
        self.max_steps = 20
        self.num_blocks = 50
        self.primitive = primitives.push

        self.lang_template = "push the pile of {block_color} blocks into the {square_color} square"
        self.task_completed_desc = "done separating pile."

    def reset(self, env):
        super().reset(env)

        color_names = self.get_colors()
        color_names = random.sample(color_names, k=3)
        zone1_color, zone2_color, block_color = [utils.COLORS[cn] for cn in color_names]

        # Add goal zone.
        zone_size = (0.15, 0.15, 0)
        zone1_pose = self.get_random_pose(env, zone_size)
        zone2_pose = self.get_random_pose(env, zone_size)
        while np.linalg.norm(np.array(zone2_pose[0]) - np.array(zone1_pose[0])) < 0.2:
            zone2_pose = self.get_random_pose(env, zone_size)

        zone1_obj_id = env.add_object('zone/zone.urdf', zone1_pose, 'fixed')
        p.changeVisualShape(zone1_obj_id, -1, rgbaColor=zone1_color + [1])
        zone2_obj_id = env.add_object('zone/zone.urdf', zone2_pose, 'fixed')
        p.changeVisualShape(zone2_obj_id, -1, rgbaColor=zone2_color + [1])

        # Choose zone
        zone_target_idx = random.randint(0, 1)
        zone_target = [zone1_pose, zone2_pose][zone_target_idx]
        zone_target_color = [color_names[0], color_names[1]][zone_target_idx]

        # Add pile of small blocks.
        obj_pts = {}
        obj_ids = []
        targets = []
        for _ in range(self.num_blocks):
            rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
            ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
            xyz = (rx, ry, 0.01)
            theta = np.random.rand() * 2 * np.pi
            xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            obj_id = env.add_object('block/small.urdf', (xyz, xyzw))
            p.changeVisualShape(obj_id, -1, rgbaColor=block_color + [1])
            obj_pts[obj_id] = self.get_box_object_points(obj_id)
            obj_ids.append((obj_id, (0, None)))
            targets.append([1, 0])

        # Goal: all small blocks must be in the correct zone zone.
        self.goals.append((obj_ids, np.ones((50, 1)), [zone_target],
                           True, False, 'zone',
                           (obj_pts, [(zone_target, zone_size)]), 1))
        self.lang_goals.append(self.lang_template.format(block_color=color_names[2],
                                                         square_color=zone_target_color))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class SeparatingPilesSeenColors(SeparatingPilesUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class SeparatingPilesFull(SeparatingPilesUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors