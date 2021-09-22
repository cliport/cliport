"""Stacking Block Pyramid Sequence task."""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p
import random

class StackBlockPyramidSeqUnseenColors(Task):
    """Stacking Block Pyramid Sequence base class."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "put the {pick} block on {place}"
        self.task_completed_desc = "done stacking block pyramid."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: make bottom row.
        self.goals.append(([objs[0]], np.ones((1, 1)), [targs[0]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[0],
                                                         place="the lightest brown block"))

        self.goals.append(([objs[1]], np.ones((1, 1)), [targs[1]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[1],
                                                         place="the middle brown block"))

        self.goals.append(([objs[2]], np.ones((1, 1)), [targs[2]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[2],
                                                         place="the darkest brown block"))

        # Goal: make middle row.
        self.goals.append(([objs[3]], np.ones((1, 1)), [targs[3]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[3],
                                                         place=f"the {color_names[0]} and {color_names[1]} blocks"))

        self.goals.append(([objs[4]], np.ones((1, 1)), [targs[4]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[4],
                                                         place=f"the {color_names[1]} and {color_names[2]} blocks"))

        # Goal: make top row.
        self.goals.append(([objs[5]], np.ones((1, 1)), [targs[5]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[5],
                                                         place=f"the {color_names[3]} and {color_names[4]} blocks"))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlockPyramidSeqSeenColors(StackBlockPyramidSeqUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class StackBlockPyramidSeqFull(StackBlockPyramidSeqUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors