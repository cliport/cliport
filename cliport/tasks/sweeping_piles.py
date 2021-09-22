"""Sweeping task."""

import numpy as np
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils


class SweepingPiles(Task):
    """Sweeping task."""

    def __init__(self):
        super().__init__()
        self.ee = Spatula
        self.max_steps = 20
        self.primitive = primitives.push
        self.lang_template = "push the pile of blocks into the green square"
        self.task_completed_desc = "done sweeping."

    def reset(self, env):
        super().reset(env)

        # Add goal zone.
        zone_size = (0.12, 0.12, 0)
        zone_pose = self.get_random_pose(env, zone_size)
        env.add_object('zone/zone.urdf', zone_pose, 'fixed')

        # Add pile of small blocks.
        obj_pts = {}
        obj_ids = []
        for _ in range(50):
            rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
            ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
            xyz = (rx, ry, 0.01)
            theta = np.random.rand() * 2 * np.pi
            xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            obj_id = env.add_object('block/small.urdf', (xyz, xyzw))
            obj_pts[obj_id] = self.get_box_object_points(obj_id)
            obj_ids.append((obj_id, (0, None)))

        # Goal: all small blocks must be in zone.
        # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
        # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
        # self.goals.append((goal, metric))
        self.goals.append((obj_ids, np.ones((50, 1)), [zone_pose], True, False,
                           'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
        self.lang_goals.append(self.lang_template)
