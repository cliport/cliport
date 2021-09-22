"""Sequential Kitting Tasks."""

import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class AssemblingKitsSeqUnseenColors(Task):
    """Sequential Kitting Tasks base class."""

    def __init__(self):
        super().__init__()
        # self.ee = 'suction'
        self.max_steps = 7
        # self.metric = 'pose'
        # self.primitive = 'pick_place'
        self.train_set = np.arange(0, 14)
        self.test_set = np.arange(14, 20)
        self.homogeneous = False

        self.lang_template = "put the {color} {obj} in the {loc}{obj} hole"
        self.task_completed_desc = "done assembling kit."

    def reset(self, env):
        super().reset(env)

        # Add kit.
        kit_size = (0.28, 0.2, 0.005)
        kit_urdf = 'kitting/kit.urdf'
        kit_pose = self.get_random_pose(env, kit_size)
        env.add_object(kit_urdf, kit_pose, 'fixed')

        # Shape Names:
        shapes = {
            0: "letter R shape",
            1: "letter A shape",
            2: "triangle",
            3: "square",
            4: "plus",
            5: "letter T shape",
            6: "diamond",
            7: "pentagon",
            8: "rectangle",
            9: "flower",
            10: "star",
            11: "circle",
            12: "letter G shape",
            13: "letter V shape",
            14: "letter E shape",
            15: "letter L shape",
            16: "ring",
            17: "hexagon",
            18: "heart",
            19: "letter M shape",
        }

        n_objects = 5
        if self.mode == 'train':
            obj_shapes = np.random.choice(self.train_set, n_objects)
        else:
            if self.homogeneous:
                obj_shapes = [np.random.choice(self.test_set)] * n_objects
            else:
                obj_shapes = np.random.choice(self.test_set, n_objects)

        color_names = self.get_colors()
        np.random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        symmetry = [
            2 * np.pi, 2 * np.pi, 2 * np.pi / 3, np.pi / 2, np.pi / 2, 2 * np.pi,
            np.pi, 2 * np.pi / 5, np.pi, np.pi / 2, 2 * np.pi / 5, 0, 2 * np.pi,
            2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 2 * np.pi / 6, 2 * np.pi, 2 * np.pi
        ]

        # Build kit.
        targets = []
        targets_spatial_desc = []
        targ_pos = [[-0.09, 0.045, 0.0014], [0, 0.045, 0.0014],
                    [0.09, 0.045, 0.0014], [-0.045, -0.045, 0.0014],
                    [0.045, -0.045, 0.0014]]
        template = 'kitting/object-template.urdf'
        for i in range(n_objects):
            shape = os.path.join(self.assets_root, 'kitting',
                                 f'{obj_shapes[i]:02d}.obj')
            scale = [0.003, 0.003, 0.0001]  # .0005
            pos = utils.apply(kit_pose, targ_pos[i])
            theta = np.random.rand() * 2 * np.pi
            rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
            urdf = self.fill_template(template, replace)
            env.add_object(urdf, (pos, rot), 'fixed')
            if os.path.exists(urdf):
                os.remove(urdf)
            targets.append((pos, rot))

            # Decide spatial description based on the location of the hole (top-down view).
            shape_type = obj_shapes[i]
            if list(obj_shapes).count(obj_shapes[i]) > 1:
                duplicate_shapes = [j for j, o in enumerate(obj_shapes) if i != j and o == shape_type]
                other_poses = [utils.apply(kit_pose, targ_pos[d]) for d in duplicate_shapes]

                if all(pos[0] < op[0] and abs(pos[0]-op[0]) > abs(pos[1]-op[1]) for op in other_poses):
                    spatial_desc = "top "
                elif all(pos[0] > op[0] and abs(pos[0]-op[0]) > abs(pos[1]-op[1]) for op in other_poses):
                    spatial_desc = "bottom "
                elif all(pos[1] < op[1] for op in other_poses):
                    spatial_desc = "left "
                elif all(pos[1] > op[1] for op in other_poses):
                    spatial_desc = "right "
                else:
                    spatial_desc = "middle "

                targets_spatial_desc.append(spatial_desc)
            else:
                targets_spatial_desc.append("")

        # Add objects.
        objects = []
        matches = []
        for i in range(n_objects):
            shape = obj_shapes[i]
            size = (0.08, 0.08, 0.02)
            pose = self.get_random_pose(env, size)
            fname = f'{shape:02d}.obj'
            fname = os.path.join(self.assets_root, 'kitting', fname)
            scale = [0.003, 0.003, 0.001]
            replace = {'FNAME': (fname,), 'SCALE': scale, 'COLOR': colors[i]}
            urdf = self.fill_template(template, replace)
            block_id = env.add_object(urdf, pose)
            if os.path.exists(urdf):
                os.remove(urdf)
            objects.append((block_id, (symmetry[shape], None)))
            match = np.zeros(len(targets))
            match[np.argwhere(obj_shapes == shape).reshape(-1)] = 1
            matches.append(match)

        target_idxs = list(range(n_objects))
        np.random.shuffle(target_idxs)
        for i in target_idxs:
            self.goals.append(([objects[i]], np.ones((1, 1)), [targets[i]],
                                False, True, 'pose', None, 1 / n_objects))
            self.lang_goals.append(self.lang_template.format(color=color_names[i],
                                                             obj=shapes[obj_shapes[i]],
                                                             loc=targets_spatial_desc[i]))
        self.max_steps = n_objects

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class AssemblingKitsSeqSeenColors(AssemblingKitsSeqUnseenColors):
    """Kitting Task - Easy variant."""

    def get_colors(self):
        return utils.TRAIN_COLORS


class AssemblingKitsSeqFull(AssemblingKitsSeqUnseenColors):
    """Kitting Task - Easy variant."""

    def __init__(self):
        super().__init__()
        self.train_set = np.arange(0, 20)
        self.test_set = np.arange(0, 20)

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors

