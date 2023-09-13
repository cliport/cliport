import os
import math
import time
import pickle
import csv
import itertools
import typing
import torch

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pylab
from matplotlib.patches import Rectangle
import matplotlib as mpl

from cliport import agents
from cliport.dataset import RealRobotDataset

from cliport.utils import utils as master_utils
from cliport.utils import utils_2 as master_utils_2

#from cliport.ros_server import RigidTransformer, get_bbox
from cliport.view_save_data import RigidTransformer
from typing import List, Tuple, Dict, Any
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class DataHandler:
    """Data reading utilities for model comparison

    Raises:
        NotImplementedError: Raised when trying to input something other than "train" or "val" to target in read_dataset.
        NotImplementedError: Raised when trying to extract from some other position than "pick" or "place" in extract_target_rot.
        NotImplementedError: Raised when trying to input something other than "train" or "val" to target in extract_target_rot.
        NotImplementedError: Raised when trying to input something other than "train" or "val" to target in get_set_limit.

    Returns:
        (object): Nothing defined
    """

    def __init__(
        self, root: str, exp_path: str, data_path: str, result_path: str, cfg_filename: str
    ) -> None:
        """Initialization method for DataHandler. Stores essential path strings & initializes class memory.

        Args:
            root (str): path pointing to cliport root (git top folder)
            exp_path (str): path pointing to exps (same exps as in cliport/train.py)
            data_path (str): path pointing to train data folder (same data as in cliport/train.py)
            result_path (str): path pointing to where results are saved
        """
        # path variables
        self.model_path = root
        self.exp_path = exp_path
        self.data_path = data_path
        self.result_path = result_path
        self.config_filename = cfg_filename
        self.checkpoint_title = ""
        self.train_count = 0
        self.validation_count = 0

        # model (pytorch)
        self.agent = None
        self.model = {}

        # data storage    @staticmethod

        self.training_dataset = None
        self.validation_dataset = None
        self.lang_goals = {}

        self.act = None

        # common hydra cfg
        try:
            self.cfg = master_utils.load_hydra_config(f"cliport/cfg/{cfg_filename}")
        except FileNotFoundError:
            # debug file path confusion
            self.cfg = master_utils.load_hydra_config(f"cliport/cliport/cfg/{cfg_filename}")

    def read_dataset(self, model_name: str, target: str) -> None:
        """Reads specified validation / training dataset (RealRobotDataset) to appropriate class memory.
        Note: DataExtractor.data_path/model_name-target must already exist.

        Args:
            model_name (str): Part of model name common to the torch model & curated dataset.
            target (str): train or val, dictating which dataset to read from.

        Raises:
            NotImplementedError: Raised when trying to input something other than "train" or "val" to target.
        """
        if target == "train":
            model_path = f"{self.data_path}/{model_name}-train"
            self.training_dataset = RealRobotDataset(
                model_path, self.cfg, n_demos=self.train_count, augment=False
            )
        elif target == "val":
            model_path = f"{self.data_path}/{model_name}-val"
            self.validation_dataset = RealRobotDataset(
                model_path, self.cfg, self.train_count, augment=False
            )
        else:
            raise NotImplementedError(
                f"Unknown data type: {target} (should be train or val)"
            )

    def set_validation_set(self, validation_set, validation_count):
        self.validation_dataset = validation_set
        self.validation_count = validation_count

    def extract_target_rot(self, index: int, mode: str, location: str, size: int=0) -> float:
        """Extracts rotation targets from dataset. Due to the nature of RealRobotDataset, this must be done
        separately from reading the main dataset in :func:`DataExtractor.read_dataset`.

        Args:
            index (int): index of data entry to extract rotation target from.
            mode (str): train, val or both, indicating which dataset to read from. Both acts on train data until
            IndexError and continues from validation data in inverse order.
            location (str): pick or place, indicating from which position data should be extracted.
            size (int): size of the dataset. Must be set if using mode 'both'

        Raises:
            NotImplementedError: Raised when trying to extract from some other position than "pick" or "place".
            NotImplementedError: Raised when trying to input something other than "train" or "val" to target.

        Returns:
            _type_ (float): Angle of rotation target in dataset
        """
        if location == "pick":
            locator = "p0_theta"
        elif location == "place":
            locator = "p1_theta"
        else:
            # TODO: there can be more positions in multitasks.
            raise NotImplementedError(f"Unknown task position: {location}")

        if mode == 'both':
            try:
                return self.training_dataset[index][0][locator]
            except IndexError:
                return self.validation_dataset[index][0][locator]
        elif mode == "train":
            return self.training_dataset[index][0][locator]
        elif mode == "val":
            return self.validation_dataset[index][0][locator]
        else:
            raise NotImplementedError(f"Unknown type: {mode} (should be train, val or both)")

    def load_model(self, model_name: str, extender: str) -> None:
        """Loads a pytorch model to class storage.

        Args:
            model_name (str): Part of model name common to the torch model & curated dataset.
            extender (str): Extender for the model name, such as '-cliport-nXX-train'. Used to test multiple models
            of same type with different amounts of demonstration data
        """
        agent = None

        # Initialize agent
        master_utils.set_seed(0, torch=True)
        agent_eval_title = "two_stream_clip_lingunet_lat_transporter"
        name = f"{model_name}-{agent_eval_title}-1-0"
        agent = agents.names[agent_eval_title](
            name, self.cfg, None, self.training_dataset
        )
        agent.load(
            f"{self.exp_path}/{model_name}{extender}/checkpoints/{self.checkpoint_title}"
        )

        self.agent = agent

    def find_latest_best_checkpoint_version(
        self, model_name: str, extender: str
    ) -> None:
        try:
            latest = None
            candidates = os.listdir(
                f"{self.exp_path}/{model_name}{extender}/checkpoints/"
            )
            candidates.sort()
            if len(candidates) > 2:
                # best-vX
                latest = candidates.pop(-3)
            elif len(candidates) > 1:
                # best
                latest = candidates.pop(-2)
            else:
                # latest (no best)
                latest = candidates[0]
            self.checkpoint_title = latest
        except FileNotFoundError as e:
            print(
                f"File not found ({e}). \
                  Does 'checkpoints' exist in {self.exp_path}{model_name}{extender}?"
            )
        except IndexError:
            print("Indexing error in finding latest best (or latest)")

    def act_on_model(
        self, obs: Dict[str, np.array], info: dict, goal: str = None
    ) -> dict:
        """Commands model to act. Model must be loaded to memory by :func:`DataHandler.load_model` before executing this
        Args:
            obs (dict[str, np.array]): image (observation) data
            info (dict): task metadata, such as lang_goal
            goal (str, optional): _description_. Defaults to None. Lang goal to execute on (by default uses
            info['lang_goal']).

        Returns:
            dict: string key dict of tuple[np.array, np.array] or list[int | float] or np.array[float] values
        """
        self.act = self.agent.act(obs, info, goal)
        return self.act

    def rot_on_model(
        self, episode: tuple, mode: str, location: str, size=0
    ) -> Tuple[Any, Any, Any]:
        """Commands model to do a rotation prediction. Model must be loaded to memory by
        :func:`DataHandler.load_model` before executing this.

        Args:
            episode (tuple): Action data for datapoint. Contains all relevant data such as image color data,
                pick & place positions, rotation data, and lang_goal.
            mode (str): 'train', 'val' or 'both', depending on which dataset should be used in the rotation process.
            location (str): either 'pick' or 'place' depending on which position the action is done on.
            size (int): size of the combined val & train datasets.

        Returns:
            tuple(np.ndarray, np.ndarray):
        """
        batch = self.get_batch(mode, episode, size)

        l = str(batch["lang_goal"])
        inp = {"inp_img": batch["img"], "lang_goal": l}
        conf = self.agent.attn_forward(inp)
        logits = conf.detach().cpu().numpy()

        if location == "pick":
            p0 = self.act["pick"]
        elif location == "place":
            p0 = self.act["place"]
        else:
            conf = conf.detach().cpu().numpy()
            argmax = np.argmax(conf)
            argmax = np.unravel_index(argmax, shape=conf.shape)
            p0 = argmax[:2]

        rot_inp = {
            "inp_img": batch["img"],
            "lang_goal": l,
            "p0": p0,
        }
        rot_conf = self.agent.attn_rot_forward(rot_inp)
        rot_conf = rot_conf.detach().cpu().numpy()

        return conf, rot_conf, logits

    def clear_model_data(self) -> None:
        """Clears relevant data from class memory for starting processing with a new model"""
        self.validation_dataset = None
        self.training_dataset = None
        self.agent = None
        self.lang_goals = {}
        self.act = None

    def augment_cfg(self, model_name: str, extender: str) -> None:
        """Adds in entries from train.yaml missing in export.yaml. This is done programmatically to gurantee right
        form for the dict.

        Args:
            model_name (str): Part of model name common to the torch model & curated dataset.
            extender (str): Extender for the model name, such as '-cliport-nXX-train' or '-cliport-nXX-val'
        """
        entry = {
            "agent": "two_stream_clip_lingunet_lat_transporter",
            "attn_stream_fusion_type": "add",
            "data_dir": self.data_path,
            "exp_folder": "exps",
            "gpu": [0],
            "lang_fusion_type": "mult",
            "load_from_last_ckpt": True,
            "log": False,
            "lr": 0.0001,
            "n_demos": self.train_count,
            "n_rotations": 36,
            "batchnorm": False,
            "n_steps": self.validation_count + self.train_count,
            "n_val": self.validation_count,
            "save_steps": [math.inf],
            "task": model_name,
            "train_dir": f"{self.exp_path}/{model_name}{extender}",
            "trans_stream_fusion_type": "conv",
            "val_repeats": 1,
        }
        self.cfg["train"] = entry

    def get_set_limit(self, model: str, mode: str) -> int:
        """Function for getting the size of validation or training datasets (read from disk, not extender of model)

        Args:
            model (str): Part of model name common to the torch model & curated dataset.
            mode (str): Either 'train' or 'val' depending on which set is being read.

        Raises:
            NotImplementedError: Raised when trying to input something other than "train" or "val" to target.

        Returns:
            (int): count of examples in training/validation dataset
        """

        try:
            count = len(os.listdir(f"{self.data_path}/{model}-{mode}"))

            if mode == "train":
                self.train_count = count
            elif mode == "val":
                self.validation_count = count
            else:
                raise NotImplementedError(f"Uknown data mode {mode}")

            return count

        except FileNotFoundError:
            print("Failed fetching training examples. Path doesn't exist")

    def get_lang_goals(self, model: str) -> Dict[str, int]:
        """Function for getting all lang goals from the model dataset (from train & val). Useful for finding typos.

        Args:
            model (str): Part of model name common to the torch model & curated dataset.

        Returns:
            list[str]: dict with keys being all unique lang_goals of the model & values their integer count
        """

        # note: using self in sub-functions might be bad practice, especially when manipulating outer context.
        def read_pkls(model, extender):

            pkls = os.listdir(f"{self.data_path}/{model}-{extender}")
            for pkl in pkls:
                with open(f"{self.data_path}/{model}-{extender}/{pkl}", "rb") as f:
                    data = pickle.load(f)
                    lang_goal = data["info"]["lang_goal"]
                    if lang_goal not in self.lang_goals:
                        self.lang_goals[lang_goal] = 1
                    else:
                        self.lang_goals[lang_goal] += 1

        read_pkls(model, "train")
        read_pkls(model, "val")
        return self.lang_goals

    def get_observation(self, index: int, mode: str, size: int=0) -> Tuple[dict, dict]:
        """Shell for reading an episode from the specified dataset.

        Args:
            index (int): index of the episode in the dataset (same as the number in the filename)
            mode (str): 'train', 'val' or 'both' depending on which set is being read. 'both' returns episodes from training set 
            until index error is reached, then in inverse from validation set (for simplicity).
            size (int): size of the read dataset. Can be unset except if mode is 'both'

        Returns: tuple: Action data for datapoint. Contains all relevant data such as image color data, pick & place
        positions, rotation data, and lang_goal.
        """
        if mode == "both":
            try:
                episode = self.training_dataset.load(index, True, False)[0][0]
            except IndexError:
                episode = self.validation_dataset.load(size - index - 1, True, False)[0][0]
        elif mode == "train":
            episode = self.training_dataset.load(index, True, False)[0][0]
        elif mode == "val":
            episode = self.validation_dataset.load(index, True, False)[0][0]
        else:
            return {}, {}
        return episode

    def organize_set(self, set):
        """de-randomizes the order in a dataset

        Args:
            set (str): Which set in self to access (either "train" or "val")

        Raises:
            NotImplementedError: Trying to access a set not understood by the class
        """
        if set == "train":
            self.training_dataset.sample_set = np.sort(self.training_dataset.sample_set)
        elif set == "val":
            self.validation_dataset.sample_set = np.sort(self.training_dataset.sample_set)
        else:
            raise NotImplementedError("Unknown dataset type.")

    def write_csv_to_disk(self, csv_text, filename: str) -> None:
        try:
            with open(f"{self.result_path}/{filename}", "w") as csvfile:
                csvfile.write(csv_text)
                """
                csv_lines = csv_text.split(';')
                for line in csv_lines:
                    items = line.split(',')
                    for item in items:
                        csvfile.write(item)
                """
        except FileNotFoundError as e:
            print(f"Error: {e}")

    def get_batch(self, mode, episode, size):
        if mode == 'both':
            try: 
                return self.training_dataset.process_sample(episode)
            except IndexError:
                return self.validation_dataset.process_sample(episode)
        elif mode == 'train':
            return self.training_dataset.process_sample(episode)
        elif mode == 'val':
            return self.validation_dataset.process_sample(episode)
        return None

    def get_images_from_episode(self, mode, episode):
        batch = self.get_batch(mode, episode)

        img = torch.from_numpy(batch["img"])
        color = np.uint8(img.detach().cpu().numpy())[:, :, :3]
        color = color.transpose(1, 0, 2)
        depth = np.array(img.detach().cpu().numpy())[:, :, 3]
        depth = depth.transpose(1, 0)

        return color, depth

    def get_points_from_episode(self, batch):
        batch = self.get_batch


class DataProcessor:
    """Utilities for processing extracted data for model comparison

    Returns:
        (object): Nothing defined
    """
    #TODO: functions in this class are regularly terrible.
    
    def __init__(self) -> None:
        """Initialization function for DataProcessor. Class contains no memory of state & most functions are
        static methods. Some functions call other class functions."""
        pass

    @staticmethod
    def find_rot_peak(prediction_data: np.array) -> float:
        argmax = np.argmax(prediction_data)
        location = np.unravel_index(argmax, shape=prediction_data.shape)
        return location[2] * (2 * np.pi / prediction_data.shape[2]) * -1.0

    @staticmethod
    def find_other_peaks(prediction_data, peak_threshold):
        # TODO: implement this
        peaks = {}
        return peaks

    @staticmethod
    def calculate_pythagorean_distance(
        target_pos: List[List[float]], predicted_pos: List[List[float]]
    ) -> List[float]:
        xd = target_pos[0][0] - predicted_pos[0][0]
        yd = target_pos[0][1] - predicted_pos[0][1]
        zd = target_pos[0][2] - predicted_pos[0][2]
        return [xd, yd, zd, math.sqrt(xd * xd + yd * yd + zd * zd)]

    def calculate_travel_error(
        self, pick_actual, place_actual, pick_prediction, place_prediction
    ):
        actual_travel = self.calculate_pythagorean_distance(place_actual, pick_actual)[
            3
        ]
        predicted_travel = self.calculate_pythagorean_distance(
            place_prediction, pick_prediction
        )[3]
        travel_error = actual_travel - predicted_travel
        return travel_error, actual_travel, predicted_travel

    @staticmethod
    def calculate_angular_distance(
        target_angle: float, predicted_angle: float
    ) -> float:
        return target_angle - predicted_angle

    @staticmethod
    def convert_dict_to_csv(
        dict_to_convert: dict, order: List[str], special_keys: list, do_box_labels: bool = False, 
    ) -> str:
        task_names = list(dict_to_convert.keys())
        if do_box_labels:
            csv_text = ","
            for task_name in task_names:
                csv_text += (
                    f"{task_name},minimum,first quartile,median,third quartile,maximum,"
                )
        else:
            csv_text = ",,"
            delim_count = np.shape(dict_to_convert[task_names[0]])[
                0
            ]  # width of data-points, assume shape
            for task_name in task_names:
                # first line containing error legends
                csv_text += f"{task_name}," + "," * (delim_count - 1)
        csv_text += "\n"
        row_entries = {}
        for task_name in task_names:
            # extracting data for rows of csv

            #height, width = np.shape(dict_to_convert[task_name]['box'])
            height = len(dict_to_convert[task_name]['box'])
            width = len(order[0])
            for i in range(width):
                error_title = order[0][i]
                if error_title in special_keys:
                    entry = dict_to_convert[task_name]['average'].pop(0)
                    if error_title not in row_entries:
                        row_entries[error_title] = [entry]
                    else:
                        row_entries[error_title].append(entry)
                    # placeholder values for proper alignment
                    for _ in range(height - 1):
                        row_entries[error_title].append(entry)
                else:
                    for k in range(height):
                        error = dict_to_convert[task_name]['box'][k][i]
                        if error_title not in row_entries:
                            row_entries[error_title] = [error]
                        else:
                            row_entries[error_title].append(error)

        for error_title in list(row_entries.keys()):
            first_entry = True
            if not do_box_labels:
                csv_text += f",{error_title},"
            for i, item in enumerate(row_entries[error_title]):
                if np.mod(i, 5) == 0 and do_box_labels:
                    if first_entry:
                        first_entry = False
                        csv_text += f",{error_title},"
                    else:
                        csv_text += f"{error_title},"
                csv_text += f"{item},"
            csv_text += "\n"
        return csv_text

    def collapse_results_to_meta_results(
        self, all_data_dict: dict, 
        use_broken_goals: bool, 
        placeholder_value: float, 
        errors_to_average: list
    ) -> (dict, list):
        # TODO: finish this
        # seems to not stash all error data.
        result_dict = {}
        all_lang_goals = {}

        model_names = list(all_data_dict.keys())

        # extract every lang_goal between all models being processed
        for model_name in model_names:
            lang_goals = all_data_dict[model_name]["lang_goals"]
            for lang_goal in lang_goals:
                if lang_goal not in all_lang_goals:
                    all_lang_goals[lang_goal] = all_data_dict[model_name]["lang_goals"][
                        lang_goal
                    ]
                else:
                    all_lang_goals[lang_goal] += all_data_dict[model_name][
                        "lang_goals"
                    ][lang_goal]

        order_stash = []
        for model_name in model_names:
            present_example_lang_goals = self.find_task_lang_goals(
                all_data_dict[model_name],
                errors_to_average
            )
            for lang_goal in all_lang_goals:
                skip_flag = False
                if lang_goal in present_example_lang_goals:
                    current_goal_data = self.find_same_tasks(
                        all_data_dict[model_name], lang_goal
                    )
                else:
                    skip_flag = True
                    current_goal_data = self.create_empty_task(placeholder_value)

                if not skip_flag or use_broken_goals:
                    """
                    (
                        averaged_goal_data,
                        average_order,
                    ) = self.calculate_average_of_errors(current_goal_data)
                    """
                    (
                        minimum_goal_data,
                        maximum_goal_data,
                        median_goal_data,
                        first_quartile_goal_data,
                        third_quartile_goal_data,
                        box_order,
                        average_goal_data,
                    ) = self.calculate_box_values_of_errors(current_goal_data, errors_to_average)
                    # reset value, doesn't seem to reset otherwise
                    boxed_goal_data = []
                    boxed_goal_data = [
                        minimum_goal_data,
                        first_quartile_goal_data,
                        median_goal_data,
                        third_quartile_goal_data,
                        maximum_goal_data,
                    ]
                    # order_stash.append(average_order)
                    order_stash.append(box_order)

                    if model_name not in result_dict:
                        result_dict[model_name] = {lang_goal: {
                            'box': boxed_goal_data, 
                            'average': average_goal_data,
                            }}
                    else:
                        result_dict[model_name][lang_goal] = {
                            'box': boxed_goal_data,
                            'average': average_goal_data,
                            }
        return result_dict, order_stash

    def collapse_results_to_results(
        self, all_data_dict: dict, placeholder_value: float
    ) -> (dict, list):
        """A simplified version of :func:`collapse_results_to_meta_results` which doesn't calculate values but uses result values "as is"
        """
        result_dict = {}
        all_lang_goals = {}

        model_names = list(all_data_dict.keys())

        # extract every lang_goal between all models being processed
        for model_name in model_names:
            lang_goals = all_data_dict[model_name]["lang_goals"]
            for lang_goal in lang_goals:
                if lang_goal not in all_lang_goals:
                    all_lang_goals[lang_goal] = all_data_dict[model_name]["lang_goals"][
                        lang_goal
                    ]
                else:
                    all_lang_goals[lang_goal] += all_data_dict[model_name][
                        "lang_goals"
                    ][lang_goal]

        order_stash = []
        for model_name in model_names:
            present_example_lang_goals = self.find_task_lang_goals(
                all_data_dict[model_name]
            )
            for lang_goal in all_lang_goals:
                if lang_goal in present_example_lang_goals:
                    current_goal_data = self.find_same_tasks(
                        all_data_dict[model_name], lang_goal
                    )
                else:
                    current_goal_data = self.create_empty_task(placeholder_value)

                all_goal_data, all_order = self.append_all_errors(current_goal_data)
                order_stash.append(all_order)

                if model_name not in result_dict:
                    result_dict[model_name] = {lang_goal: all_goal_data}
                else:
                    result_dict[model_name][lang_goal] = all_goal_data
        return result_dict, order_stash

    @staticmethod
    def find_same_tasks(data_dict: dict, lang_goal) -> list:
        string_values = ['lang_goals', 'math_pick_success', 'math_place_success', 'user_pick_validation', 'user_place_validation']
        return [
            data_dict[entry][0]
            for entry in data_dict
            if str(entry) not in string_values and data_dict[entry][1] == lang_goal
        ]

    @staticmethod
    def create_empty_task(value: float = 0) -> list:
        # TODO: -1 values should be #N/A in csv in order to not pollute real data.
        #  Breaks :func:`DataProcessor.calculate_average_of_errors`
        return [
            {
                "pick_x_error": value,
                "pick_y_error": value,
                "pick_z_error": value,
                "total_pick_error": value,
                "place_x_error": value,
                "place_y_error": value,
                "place_z_error": value,
                "total_place_error": value,
                "pick_rotation_error": value,
                "place_rotation_error": value,
                "travel_error": value,
                "actual_travel": value,
                "predicted_travel": value,
            }
        ]

    @staticmethod
    def calculate_average_of_errors(dict_list: List[dict]):
        error_keys = list(dict_list[0].keys())
        result = []
        order = []
        for error_key in error_keys:
            result.append(sum(item[error_key] for item in dict_list) / len(dict_list))
            order.append(error_key)
        return result, order

    @staticmethod
    def calculate_box_values_of_errors(dict_list: List[dict], special_entries: list):
        error_keys = list(dict_list[0].keys())
        order = []
        minimum = []
        maximum = []
        median = []
        first_quartile = []
        third_quartile = []
        average = []
        # across every model (dict list item), get values with error_key (x_error etc.) & find box values. Since this
        # could be random, create an "order" list as well (current main checks only the first value in
        # convert_dict_to_csv)
        for error_key in error_keys:

            order.append(error_key)
            values = [item[error_key] for item in dict_list]
            if error_key in special_entries:
                average.append(np.average(values))
            else:
                minimum.append(min(values))
                maximum.append(max(values))
                median.append(np.median(values))
                first_quartile.append(np.percentile(values, 25))
                third_quartile.append(np.percentile(values, 75))

        return minimum, maximum, median, first_quartile, third_quartile, order, average

    @staticmethod
    def append_all_errors(dict_list: List[dict]):
        error_keys = list(dict_list[0].keys())
        order = []
        all_data = []
        for error_key in error_keys:
            order.append(error_key)
            values = [item[error_key] for item in dict_list]
            all_data.append(values)

        return all_data, order

    @staticmethod
    def find_task_lang_goals(data_dict: dict, special_entries: list) -> list:
        goal_list = []

        for entry in data_dict:
            if entry not in special_entries and entry != "lang_goals":
                # lang_goals might not have any entries (should be impossible), this should be separate
                if data_dict[entry][1] not in goal_list:
                    goal_list.append(data_dict[entry][1])
        return goal_list

    @staticmethod
    def extract_rot_theta_from_pred(rot_conf):
        argmax = np.argmax(rot_conf)
        argmax = np.unravel_index(argmax, shape=rot_conf.shape)
        return argmax


    @staticmethod
    def point_inside_polygon(poly_corners, real_point):
        point = Point(real_point)
        polygon = Polygon(poly_corners)
        
        return polygon.contains(point)

    @staticmethod
    def get_angled_rectangle_corners_from_centerpoint(x, y, w, h, theta):
        """this function was made by ChatGPT

        Args:
            x (float): x coordinate of rectangle middlepoint
            y (float): y coordinate of rectangle middlepoint
            w (float): width of rectangle (x dir)
            h (float): height of rectangle (y dir)
            theta (float): angle of rectangle

        Returns:
            List: list of coordinate tuples for the rotated rectangle corners
        """
        half_w = w/2
        half_h = w/2
        
        corners = [
            (-half_w, -half_h),
            (-half_w, half_h),
            (half_w, half_h),
            (half_w, -half_h)
        ]
        
        # translate rotated corners to actual coordinates
        rotated_corners = []
        for corner in corners:
            x_corner, y_corner = corner
            x_rotated = x_corner * np.cos(theta) - y_corner * np.sin(theta)
            y_rotated = x_corner * np.sin(theta) + y_corner * np.cos(theta)
            rotated_corners.append((x_rotated, y_rotated))
            
        return [(x + xr, y + yr) for xr, yr in rotated_corners]

class DataDrawer:
    """Collection of bespoke functions for drawing data. 
    """
    def __init__(
            self, 
            admission_pick_rectangle_width=0, 
            admission_pick_rectangle_height=0,
            admission_place_rectangle_dimension=0,
            rows=1, 
            cols=1, 
            wrats=None, 
            hrats=None
        ) -> None:
        if rows != 1 or cols != 1:
            self.fig, self.axslist = plt.subplots(
                nrows=rows,
                ncols=cols,
                squeeze=True,
                width_ratios=wrats,
                height_ratios=hrats,
                subplot_kw=None,
                gridspec_kw=None,
                constrained_layout=False,
            )
            self.axs = self.axslist[0][0]
                    
            for iy, ix in np.ndindex(self.axslist.shape):
                axs = self.axslist[iy][ix]
                axs.axes.xaxis.set_visible(False)
                axs.axes.yaxis.set_visible(False)
        else:
            self.fig, self.axs = plt.subplots(
                nrows=rows,
                ncols=cols,
                squeeze=True,
                width_ratios=wrats,
                height_ratios=hrats,
                subplot_kw=None,
                gridspec_kw=None,
                constrained_layout=False,
            )
            self.axs.axes.xaxis.set_visible(False)
            self.axs.axes.yaxis.set_visible(False)
        self.init = True
        plt.ion()
        self.fig.suptitle("prediction cast pick/place (blue/cyan) vs. actual (red/pink)")

        # action values
        self.pick_predict = None
        self.pick_actual = None
        self.place_predict = None
        self.place_actual = None
        self.pick_rot_predict = None
        self.pick_rot_actual = None
        self.place_rot_actual = None
        self.place_rot_predict = None
        
        self.admission_pick_rectangle_width = admission_pick_rectangle_width
        self.admission_pick_rectangle_height = admission_pick_rectangle_height
        self.admission_place_rectangle_dimension = admission_place_rectangle_dimension

        self.rigid_transformer = RigidTransformer()

    def set_action_values(
        self,
        pick_pred,
        pick_act,
        place_pred,
        place_act,
        pick_rot_pred,
        pick_rot_act,
        place_rot_pred,
        place_rot_act,
        title
    ):
        self.pick_predict = pick_pred
        self.pick_actual = pick_act
        self.place_predict = place_pred
        self.place_actual = place_act
        self.pick_rot_predict = pick_rot_pred
        self.pick_rot_actual = pick_rot_act
        self.place_rot_predict = place_rot_pred
        self.place_rot_actual = place_rot_act

        self.fig.canvas.manager.set_window_title(title)

    def get_cast_admission_rectangle_values(self, x_pick_pred, y_pick_pred, x_place_pred, y_place_pred):
        cast_pick_width_px = self.rigid_transformer.xyz_to_pix(
        [
            self.pick_predict[0][0] + self.admission_pick_rectangle_width, 
            self.pick_predict[0][1], 
            self.pick_predict[0][2]
        ]
        )
        cast_pick_heigth_px = self.rigid_transformer.xyz_to_pix(
        [
            self.pick_predict[0][0], 
            self.pick_predict[0][1] + self.admission_pick_rectangle_height, 
            self.pick_predict[0][2]
        ]
        )
        cast_place_dim_px = self.rigid_transformer.xyz_to_pix(
        [
            self.place_predict[0][0] + self.admission_place_rectangle_dimension,
            self.place_predict[0][1] + self.admission_place_rectangle_dimension,
            self.place_predict[0][2]
        ]
        )
        xdpiw = x_pick_pred - cast_pick_width_px[0]
        ydpiw = y_pick_pred - cast_pick_width_px[1]
        xdpih = x_pick_pred - cast_pick_heigth_px[0]
        ydpih = y_pick_pred - cast_pick_heigth_px[1]
        xdplw = x_place_pred - cast_place_dim_px[0]
        ydplh = y_place_pred - cast_place_dim_px[1]
        pick_adm_rect_width_cast = math.sqrt(math.pow(xdpiw, 2) + math.pow(ydpiw, 2))
        pick_adm_rect_height_cast = math.sqrt(math.pow(xdpih, 2) + math.pow(ydpih, 2))
        place_adm_rect_dim_cast = math.sqrt(math.pow(xdplw, 2) + math.pow(ydplh, 2))
        
        return pick_adm_rect_width_cast, pick_adm_rect_height_cast, place_adm_rect_dim_cast

    def add_admission_rectangles(self, x_pick_pred, y_pick_pred, x_place_pred, y_place_pred):
        # pick admission rectangle
        (
            pick_adm_rect_width_cast, 
            pick_adm_rect_height_cast, 
            place_adm_rect_dim_cast 
        ) = self.get_cast_admission_rectangle_values(
            x_pick_pred, y_pick_pred, x_place_pred, y_place_pred 
        )
        pick_adm_rect = Rectangle(
                (
                    x_pick_pred - pick_adm_rect_width_cast/2, 
                    y_pick_pred - pick_adm_rect_height_cast/2
                ), 
                pick_adm_rect_width_cast,
                pick_adm_rect_height_cast,
                0,
                edgecolor = 'red',
                facecolor = 'none',
                lw = 2,
                zorder = 2,
                )
        place_adm_rect = Rectangle(
                (
                    x_place_pred - place_adm_rect_dim_cast/2, 
                    y_place_pred - place_adm_rect_dim_cast/2
                ), 
                place_adm_rect_dim_cast,
                place_adm_rect_dim_cast,
                0,
                edgecolor = 'pink',
                facecolor = 'none',
                lw = 2,
                zorder = 2,
                )
        
        # apply rotation
        pick_rot_transform = mpl.transforms.Affine2D().rotate_around(
            x_pick_pred, y_pick_pred, self.pick_rot_predict) + self.axs.transData
        place_rot_transform = mpl.transforms.Affine2D().rotate_around(
            x_place_pred, y_place_pred, self.place_rot_predict) + self.axs.transData
        
        pick_adm_rect.set_transform(pick_rot_transform)
        place_adm_rect.set_transform(place_rot_transform)
        
        self.axs.add_patch(pick_adm_rect)
        self.axs.add_patch(place_adm_rect)

    def draw_im_data(self, im_data, use_predicted_data, draw_admission_rectangles):
        self.axs.cla()
        if self.init:
            plt.sca(self.axs)
        self.axs.imshow(im_data, animated=True)

        if use_predicted_data:
            self.draw_predicted_data(draw_admission_rectangles)
        
        self.fig.canvas.draw()

        plt.pause(0.02)
        self.fig.canvas.flush_events()

    def draw_predicted_data(self, draw_admission_rectangles):
        # plot predicted data
        x_pick_pred, y_pick_pred = self.rigid_transformer.xyz_to_pix(self.pick_predict[0])
        x_place_pred, y_place_pred = self.rigid_transformer.xyz_to_pix(self.place_predict[0])
        self.axs.plot(x_pick_pred, y_pick_pred, marker="o", markeredgecolor="red", markerfacecolor="red")
        self.axs.plot(x_place_pred, y_place_pred, marker="o", markeredgecolor="pink", markerfacecolor="pink")
        self.draw_grasp_lines([x_pick_pred, y_pick_pred], self.pick_rot_predict, "red")
        self.draw_grasp_lines([x_place_pred, y_place_pred], self.place_rot_predict, "pink")

        # plot actual data
        x_pick_act, y_pick_act = self.rigid_transformer.xyz_to_pix(self.pick_actual[0])
        x_place_act, y_place_act = self.rigid_transformer.xyz_to_pix(self.place_actual[0])
        self.axs.plot(x_pick_act, y_pick_act, marker="o", markeredgecolor="blue", markerfacecolor="blue")
        self.axs.plot(x_place_act, y_place_act, marker="o", markeredgecolor="cyan", markerfacecolor="cyan")
        self.draw_grasp_lines([x_pick_act, y_pick_act], self.pick_rot_actual, "blue")
        self.draw_grasp_lines([x_place_act, y_place_act], self.place_rot_actual, "cyan")

        if draw_admission_rectangles:
            self.add_admission_rectangles(x_pick_pred, y_pick_pred, x_place_pred, y_place_pred)

    def draw_data_to_active_axs(self, data):
        self.axs.cla()
        self.axs.imshow(data, animated=True)
        self.fig.canvas.draw()
        plt.pause(0.02)
        self.fig.canvas.flush_events()

    def draw_grasp_lines(self, position, theta, plot_color):
        line_len = 100
        pick0 = (
            position[0] + line_len / 2.0 * np.sin(theta),
            position[1] + line_len / 2.0 * np.cos(theta),
        )
        pick1 = (
            position[0] - line_len / 2.0 * np.sin(theta),
            position[1] - line_len / 2.0 * np.cos(theta),
        )
        self.axs.plot(
            (pick1[0], pick0[0]), (pick1[1], pick0[1]), color=plot_color, linewidth=1
        )

    @staticmethod
    def get_grasp_line_pix(position, theta, line_len=20):
        pos0 = (
            int(np.round(position[0] + line_len / 2.0 * np.sin(theta))),
            int(np.round(position[1] + line_len / 2.0 * np.cos(theta))),
        )
        pos1 = (
            int(np.round(position[0] - line_len / 2.0 * np.sin(theta))),
            int(np.round(position[1] - line_len / 2.0 * np.cos(theta))),
        )
        return (pos0, pos1)

    def set_axs(self, r, c):
        self.axs = self.axslist[r][c]
