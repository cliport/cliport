"""Packing Google Objects tasks."""

import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p


class PackingSeenGoogleObjectsSeq(Task):
    """Packing Seen Google Objects Group base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "pack the {obj} in the brown box"
        self.task_completed_desc = "done packing objects."
        self.object_names = self.get_object_names()

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'screwdriver',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
            'val': [
                'alarm clock',
                'android toy',
                'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'screwdriver',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
            'test': [
                'alarm clock',
                'android toy',
                'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'screwdriver',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
        }

    def reset(self, env):
        super().reset(env)

        # object names
        object_names = self.object_names[self.mode]

        # Add container box.
        zone_size = self.get_random_size(0.2, 0.35, 0.2, 0.35, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf): os.remove(container_urdf)

        margin = 0.01
        min_object_dim = 0.08
        bboxes = []

        # Construct K-D Tree to roughly estimate how many objects can fit inside the box.
        # TODO(Mohit): avoid building K-D Trees
        class TreeNode:

            def __init__(self, parent, children, bbox):
                self.parent = parent
                self.children = children
                self.bbox = bbox  # min x, min y, min z, max x, max y, max z

        def KDTree(node):
            size = node.bbox[3:] - node.bbox[:3]

            # Choose which axis to split.
            split = size > 2 * min_object_dim
            if np.sum(split) == 0:
                bboxes.append(node.bbox)
                return
            split = np.float32(split) / np.sum(split)
            split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

            # Split along chosen axis and create 2 children
            cut_ind = np.random.rand() * \
                      (size[split_axis] - 2 * min_object_dim) + \
                      node.bbox[split_axis] + min_object_dim
            child1_bbox = node.bbox.copy()
            child1_bbox[3 + split_axis] = cut_ind - margin / 2.
            child2_bbox = node.bbox.copy()
            child2_bbox[split_axis] = cut_ind + margin / 2.
            node.children = [
                TreeNode(node, [], bbox=child1_bbox),
                TreeNode(node, [], bbox=child2_bbox)
            ]
            KDTree(node.children[0])
            KDTree(node.children[1])

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = TreeNode(None, [], bbox=np.array(root_size))
        KDTree(root)

        # Add Google Scanned Objects to scene.
        object_points = {}
        object_ids = []
        bboxes = np.array(bboxes)
        scale_factor = 5
        object_template = 'google/object-template.urdf'
        chosen_objs, repeat_category = self.choose_objects(object_names, len(bboxes))
        object_descs = []
        for i, bbox in enumerate(bboxes):
            size = bbox[3:] - bbox[:3]
            max_size = size.max()
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2
            shape_size = max_size * scale_factor
            pose = self.get_random_pose(env, size)

            # Add object only if valid pose found.
            if pose[0] is not None:
                # Initialize with a slightly tilted pose so that the objects aren't always erect.
                slight_tilt = utils.q_mult(pose[1], (-0.1736482, 0, 0, 0.9848078))
                ps = ((pose[0][0], pose[0][1], pose[0][2]+0.05), slight_tilt)

                object_name = chosen_objs[i]
                object_name_with_underscore = object_name.replace(" ", "_")
                mesh_file = os.path.join(self.assets_root,
                                         'google',
                                         'meshes_fixed',
                                         f'{object_name_with_underscore}.obj')
                texture_file = os.path.join(self.assets_root,
                                            'google',
                                            'textures',
                                            f'{object_name_with_underscore}.png')

                try:
                    replace = {'FNAME': (mesh_file,),
                               'SCALE': [shape_size, shape_size, shape_size],
                               'COLOR': (0.2, 0.2, 0.2)}
                    urdf = self.fill_template(object_template, replace)
                    box_id = env.add_object(urdf, ps)
                    if os.path.exists(urdf):
                        os.remove(urdf)
                    object_ids.append((box_id, (0, None)))

                    texture_id = p.loadTexture(texture_file)
                    p.changeVisualShape(box_id, -1, textureUniqueId=texture_id)
                    p.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1])
                    object_points[box_id] = self.get_mesh_object_points(box_id)

                    object_descs.append(object_name)
                except Exception as e:
                    print("Failed to load Google Scanned Object in PyBullet")
                    print(object_name_with_underscore, mesh_file, texture_file)
                    print(f"Exception: {e}")

        self.set_goals(object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size)

        for i in range(480):
            p.stepSimulation()

    def choose_objects(self, object_names, k):
        repeat_category = None
        return np.random.choice(object_names, k, replace=False), repeat_category

    def set_goals(self, object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size):
        # Random picking sequence.
        num_pack_objs = np.random.randint(1, len(object_ids))

        object_ids = object_ids[:num_pack_objs]
        true_poses = []
        for obj_idx, (object_id, _) in enumerate(object_ids):
            true_poses.append(zone_pose)

            chosen_obj_pts = dict()
            chosen_obj_pts[object_id] = object_points[object_id]

            self.goals.append(([(object_id, (0, None))], np.int32([[1]]), [zone_pose],
                               False, True, 'zone',
                               (chosen_obj_pts, [(zone_pose, zone_size)]),
                               1 / len(object_ids)))
            self.lang_goals.append(self.lang_template.format(obj=object_descs[obj_idx]))

        # Only mistake allowed.
        self.max_steps = len(object_ids)+1


class PackingUnseenGoogleObjectsSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with orange stripes',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hard drive',
                'honey dipper',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'porcelain cup',
                'purple tape',
                'red and white flashlight',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
            ],
            'val': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'dinosaur figure',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
            'test': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'dinosaur figure',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
        }


class PackingSeenGoogleObjectsGroup(PackingSeenGoogleObjectsSeq):
    """Packing Seen Google Objects Group task."""

    def __init__(self):
        super().__init__()
        self.lang_template = "pack all the {obj} objects in the brown box"
        self.max_steps = 3

    def choose_objects(self, object_names, k):
        # Randomly choose a category to repeat.
        chosen_objects = np.random.choice(object_names, k, replace=True)
        repeat_category, distractor_category = np.random.choice(chosen_objects, 2, replace=False)
        num_repeats = np.random.randint(2, 3)
        chosen_objects[:num_repeats] = repeat_category
        chosen_objects[num_repeats:2*num_repeats] = distractor_category

        return chosen_objects, repeat_category

    def set_goals(self, object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size):
        # Pack all objects of the chosen (repeat) category.
        num_pack_objs = object_descs.count(repeat_category)
        true_poses = []

        chosen_obj_pts = dict()
        chosen_obj_ids = []
        for obj_idx, (object_id, info) in enumerate(object_ids):
            if object_descs[obj_idx] == repeat_category:
                true_poses.append(zone_pose)
                chosen_obj_pts[object_id] = object_points[object_id]
                chosen_obj_ids.append((object_id, info))

        self.goals.append((
            chosen_obj_ids, np.eye(len(chosen_obj_ids)), true_poses, False, True, 'zone',
            (chosen_obj_pts, [(zone_pose, zone_size)]), 1))
        self.lang_goals.append(self.lang_template.format(obj=repeat_category))

        # Only one mistake allowed.
        self.max_steps = num_pack_objs+1


class PackingUnseenGoogleObjectsGroup(PackingSeenGoogleObjectsGroup):
    """Packing Unseen Google Objects Group task."""

    def __init__(self):
        super().__init__()

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with orange stripes',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hard drive',
                'honey dipper',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'porcelain cup',
                'purple tape',
                'red and white flashlight',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
            ],
            'val': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'dinosaur figure',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
            'test': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'dinosaur figure',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
        }