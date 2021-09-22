"""Towers of Hanoi task."""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p
import random

class TowersOfHanoiSeqUnseenColors(Task):
    """Towers of Hanoi Sequence base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 14
        self.lang_template = "move the {obj} ring to the {loc}"
        self.task_completed_desc = "solved towers of hanoi."

    def reset(self, env):
        super().reset(env)

        # Add stand.
        base_size = (0.12, 0.36, 0.01)
        base_urdf = 'hanoi/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # All colors.
        full_color_names = self.get_colors()

        # Choose three colors for three rings.
        color_names = random.sample(full_color_names, 3)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Rod positions in base coordinates.
        rod_pos = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))
        rod_names = ('lighter brown side', 'middle of the stand', 'darker brown side')

        # Add disks.
        disks = []
        disks_names = {}
        n_disks = 3
        for i in range(n_disks):
            disk_urdf = 'hanoi/disk%d.urdf' % i
            pos = utils.apply(base_pose, rod_pos[0])
            z = 0.015 * (n_disks - i - 2)
            pos = (pos[0], pos[1], pos[2] + z)
            ring_id = env.add_object(disk_urdf, (pos, base_pose[1]))
            p.changeVisualShape(ring_id, -1, rgbaColor=colors[i] + [1])
            disks.append(ring_id)
            disks_names[ring_id] = color_names[i]

        # Solve Hanoi sequence with dynamic programming.
        hanoi_steps = []  # [[object index, from rod, to rod], ...]

        def solve_hanoi(n, t0, t1, t2):
            if n == 0:
                hanoi_steps.append([n, t0, t1])
                return
            solve_hanoi(n - 1, t0, t2, t1)
            hanoi_steps.append([n, t0, t1])
            solve_hanoi(n - 1, t2, t1, t0)

        solve_hanoi(n_disks - 1, 0, 2, 1)

        # Goal: pick and place disks using Hanoi sequence.
        for step in hanoi_steps:
            disk_id = disks[step[0]]
            targ_pos = rod_pos[step[2]]
            targ_pos = utils.apply(base_pose, targ_pos)
            targ_pose = (targ_pos, (0, 0, 0, 1))
            self.goals.append(([(disk_id, (0, None))], np.int32([[1]]), [targ_pose],
                               False, True, 'pose', None, 1 / len(hanoi_steps)))
            self.lang_goals.append(self.lang_template.format(obj=disks_names[disk_id],
                                                             loc=rod_names[step[2]]))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class TowersOfHanoiSeqSeenColors(TowersOfHanoiSeqUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class TowersOfHanoiSeqFull(TowersOfHanoiSeqUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors