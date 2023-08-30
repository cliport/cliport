import hydra
import os
from omegaconf import DictConfig, OmegaConf

from cliport.model_comparison.comparison_utilities import DataHandler, DataProcessor, DataDrawer
from typing import List

ROOT_DIR = "/home/opendr/mikael/cliport"
RESULT_FOLDER = f"{ROOT_DIR}/comparison_results"
DATA_FOLDER = f"{ROOT_DIR}/data"
EXP_FOLDER = f"{ROOT_DIR}/exps"
MODELS = [
    "engine-parts-to-box-single-list",
    "engine-parts-single",
]
# extenders are required because of requirement to test the same model with
# multiple training examples. These extenders could be programmatically found as well.
# (mainly because previous project results are compared to current project results / data pollution)
MODEL_EXTENDERS = [
    "-cliport-n88-train",
    "-cliport-n34-train",
]
MODE = "single"
TYPE = "common"
SET = "val"
# Create placeholder (empty) data for validation goals that have no data (typically due to misspelling)
USE_BROKEN_GOALS = False
PLACEHOLDER_VALUE = -1.0
# Create and use amalgamation dataset for validation (amalgam must be created manually)
USE_AMALGAM = True
AMALGAM_TITLE = "complete-unseen-amalgamation"

WRITE_INDIVIDUAL_CSVS = False

# write just box values (5 values) or ALL values
CALCULATE_BOX_VALUES = True
USE_ABS_VALUES = True


def main() -> int:
    """_summary_
      main program loop

    Args:

      user actions before execution (make edits to extraction.yaml):
        1. describe location for models
        2. describe location for training data
        3. list names of models to load
        4. choose how many actions to make for each lang goal
        5. choose mode (compare models or compare stages of single model) (? could be aggregated)

      # TODO: 1. describe location for models
      # TODO: 2. describe location for training data
      # TODO: 3. list names of models to load & model name extenders
      # TODO: 4. choose mode (compare models (single) or compare stages of single model (stage))
      # TODO: 5. choose type of comparison (common -> compare common language goals, all -> get results for every lang goal present)
      # TODO: consider aggregating stage & models comparison
    """

    data_extractor = DataHandler(
        root=ROOT_DIR,
        exp_path=EXP_FOLDER,
        data_path=DATA_FOLDER,
        result_path=RESULT_FOLDER,
    )
    data_processor = DataProcessor()

    data_drawer = DataDrawer()
    
    all_data_dict = {}

    subdict = {}
    for i in range(len(MODELS)):
        subdict = {}    # empty assignment is necessary or subdict will reference the same variable for all models
        validation_size, training_size, lang_goals = read_model(data_extractor, i)
        size = training_size if SET == "train" else validation_size

        for index in range(size):
            subdict[index] = calculate_values(data_extractor, data_processor, data_drawer, size, index)
            
            episode = data_extractor.get_observation(index, SET)
            (obs, act_actual, _, info) = episode
            data_drawer.draw_im_data(obs)

        subdict["lang_goals"] = lang_goals
        all_data_dict[f"{MODELS[i]}{MODEL_EXTENDERS[i]}"] = subdict

        # clear models & agent from memory
        data_extractor.clear_model_data()

    if CALCULATE_BOX_VALUES:
        # collapse dict so tasks are not present in multiples but as statistical values
        result_dict, order = data_processor.collapse_results_to_meta_results(
            all_data_dict, USE_BROKEN_GOALS, PLACEHOLDER_VALUE
        )
    else:
        result_dict, order = data_processor.collapse_results_to_results(
            all_data_dict, PLACEHOLDER_VALUE
        )

    save_data(data_extractor, data_processor, result_dict, order)

    # free up rogue memory (?)
    del data_extractor
    print("Data extraction complete")
    return 0


def read_model(data_extractor: DataHandler, index: int):
    model = MODELS[index]
    extender = MODEL_EXTENDERS[index]
    print(f"\nReading model: {model} ({index+1}/{len(MODELS)})\n")
    # read values from directories (data storage inside extractor)
    training_size = data_extractor.get_set_limit(model, "train")
    if not USE_AMALGAM:
        validation_size = data_extractor.get_set_limit(model, "val")
        data_extractor.read_dataset(model, "val")
    else:
        validation_size = data_extractor.get_set_limit(AMALGAM_TITLE, "val")
        data_extractor.read_dataset(AMALGAM_TITLE, "val")
        
    lang_goals = data_extractor.get_lang_goals(model)
    
    data_extractor.read_dataset(model, "train")
    data_extractor.find_latest_best_checkpoint_version(
        model, extender
    )
    data_extractor.augment_cfg(model, extender)
    data_extractor.load_model(model, extender)

    return validation_size, training_size, lang_goals


def calculate_values(data_extractor: DataHandler, data_processor: DataProcessor, data_drawer: DataDrawer, size: int, index: int):
    print(f"Task {index+1}/{size}")
    episode = data_extractor.get_observation(index, SET)
    (obs, act_actual, _, info) = episode
    # goal is pulled from info if not specified
    act_prediction = data_extractor.act_on_model(obs, info, goal=None)

    # extract positional values
    pick_actual = act_actual["pose0"]
    place_actual = act_actual["pose1"]
    pick_prediction = act_prediction["pose0"]
    place_prediction = act_prediction["pose1"]

    # extract rotational values
    ert = data_extractor.extract_target_rot
    pick_rot_actual = ert(index, SET, "pick")
    place_rot_actual = ert(index, SET, "place")

    rom = data_extractor.rot_on_model
    (
        pick_conf,
        pick_rot_pred_conf,
        pick_logits,
    ) = rom(episode, SET, "pick")
    place_conf, place_rot_pred_conf, place_logits = rom(episode, SET, "place")

    frp = data_processor.find_rot_peak
    pick_rot_prediction = frp(pick_rot_pred_conf)
    place_rot_prediction = frp(place_rot_pred_conf)

    # send values to plotter
    data_drawer.set_action_values(
        pick_prediction, 
        pick_actual, 
        place_prediction, 
        place_actual, 
        pick_rot_prediction, 
        pick_rot_actual, 
        place_rot_prediction, 
        place_rot_actual
        )

    # calculate errors
    cpd = data_processor.calculate_pythagorean_distance
    pick_dist_err_collection = cpd(pick_actual, pick_prediction)
    place_dist_err_collection = cpd(place_actual, place_prediction)

    cad = data_processor.calculate_angular_distance
    pick_rot_err = cad(pick_rot_actual, pick_rot_prediction)
    place_rot_err = cad(place_rot_actual, place_rot_prediction)

    travel_errors = data_processor.calculate_travel_error(
        pick_actual, place_actual, pick_prediction, place_prediction
    )

    del ert, rom, frp, cpd, cad

    if USE_ABS_VALUES:
        return [
                {
                    "pick_x_error": abs(pick_dist_err_collection[0]),
                    "pick_y_error": abs(pick_dist_err_collection[1]),
                    "pick_z_error": abs(pick_dist_err_collection[2]),
                    "total_pick_error": abs(pick_dist_err_collection[3]),
                    "place_x_error": abs(place_dist_err_collection[0]),
                    "place_y_error": abs(place_dist_err_collection[1]),
                    "place_z_error": abs(place_dist_err_collection[2]),
                    "total_place_error": abs(place_dist_err_collection[3]),
                    "pick_rotation_error": abs(pick_rot_err),
                    "place_rotation_error": abs(place_rot_err),
                    "travel_error": abs(travel_errors[0]),
                    "actual_travel": abs(travel_errors[1]),
                    "predicted_travel": abs(travel_errors[2]),
                },
                info["lang_goal"],
            ]
    else:
        return [
                {
                    "pick_x_error": pick_dist_err_collection[0],
                    "pick_y_error": pick_dist_err_collection[1],
                    "pick_z_error": pick_dist_err_collection[2],
                    "total_pick_error": pick_dist_err_collection[3],
                    "place_x_error": place_dist_err_collection[0],
                    "place_y_error": place_dist_err_collection[1],
                    "place_z_error": place_dist_err_collection[2],
                    "total_place_error": place_dist_err_collection[3],
                    "pick_rotation_error": pick_rot_err,
                    "place_rotation_error": place_rot_err,
                    "travel_error": travel_errors[0],
                    "actual_travel": travel_errors[1],
                    "predicted_travel": travel_errors[2],
                },
                info["lang_goal"],
            ]


def save_data(data_extractor: DataHandler, data_processor: DataProcessor, all_data_dict: dict, order: List[str]):
    blurb_text = ""
    for model in all_data_dict:
        csv_text = data_processor.convert_dict_to_csv(all_data_dict[model], order, True)
        if WRITE_INDIVIDUAL_CSVS:
            data_extractor.write_csv_to_disk(csv_text, f"{model}.csv")
        elif blurb_text == "":
            blurb_text = model + "\n" + csv_text + "\n"
        else:
            blurb_text = blurb_text + "\n" + model + "\n" + csv_text + "\n"

    if not WRITE_INDIVIDUAL_CSVS:
        data_extractor.write_csv_to_disk(blurb_text, "blurb.csv")


if __name__ == "__main__":
    os.environ["CLIPORT_ROOT"] = "/home/drubuntu/cliport/cliport"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    main()
