import hydra
import os
from omegaconf import DictConfig, OmegaConf

from cliport.model_comparison.comparison_utilities import DataHandler, DataProcessor, DataDrawer
from typing import List

ROOT_DIR = "/home/drubuntu/cliport"

RESULT_FOLDER = f"{ROOT_DIR}/comparison_results"
DATA_FOLDER = f"{ROOT_DIR}/data"
EXP_FOLDER = f"{ROOT_DIR}/exps"
CFG_FILE = "train.yaml"
MODELS = [
    "engine-parts-to-box-single-list",
    "engine-parts-multi",
    "packing-objects",
]
# extenders are required because of requirement to test the same model with
# multiple training examples. These extenders could be programmatically found as well.
# (mainly because previous project results are compared to current project results / data pollution)
MODEL_EXTENDERS = [
    "-cliport-n88-train",
    "-cliport-n39-train",
    "-cliport-n119-train"
]
ALIASES = [
    {
        'xput rocker arm in red box': 'put rocker arm in red box',
        "'put push rod in brown box": 'put push rod in brown box',
        '12hput rocker arm in brown box': 'put rocker arm in brown box',
        'put long screw in brown box': 'put bolt in brown box',
        'put long screw in red box': 'put bolt in red box',
    },
    {
        'put bolt in brown box': 'put all long screws in brown box',
        'put bolt in red box': 'put all long screws in red box',
        'put push rod in red box': 'put all push rods in red box',
        'put push rod in brown box': 'put all push rods in brown box',
        'hput all long screws in brown box': 'put all long screws in brown box',
    },
    {
        'put bolt in brown box': 'put long screw in brown box',
        'hhput long screw in brown box': 'put long screw in brown box',
        'xput long screw in brown box': 'put long screw in brown box',
        'hput long screw in brown box': 'put long screw in brown box'
    }
]
# Unused
MODE = "single"
# Unused
TYPE = "common"

# Set can be "train", "val" or "both" depending on which sets are wanted. 
# Note that "both" must have both "train" and "val" sets present.
SET = "val"
# Create placeholder (empty) data for validation goals that have no data (typically due to misspelling)
USE_BROKEN_GOALS = False
PLACEHOLDER_VALUE = -1.0
# Create and use amalgamation dataset for validation (amalgam must be created manually)
USE_AMALGAM = True
AMALGAM_TITLE = "complete-unseen-amalgamation"

# Write inputs into separate csv files for all models or create a single dump csv
WRITE_INDIVIDUAL_CSVS = False

# write just box values (5 values) or ALL values
CALCULATE_BOX_VALUES = True
USE_ABS_VALUES = True

ASK_USER_VALIDATION = True

# possiblity to skip asking for model to act on the data when this is false (used for dataset inspection)
DO_PREDICTION = True

# calculate success mathematically (not accurate for multiple same objects)
DO_EVAL_SUCCESS_MATH = True
ADMITTED_PICK_WIDTH = 0.081
ADMITTED_PICK_HEIGHT = 0.15
ADMITTED_PLACE_DIM = 0.1

# Add box labels to CSV (median, quartiles)
DO_BOX_LABELS = True

# list of error keys that should be averaged instead of box valued (median, useful for bools)
ERRORS_TO_AVERAGE  = [
    'math_pick_success', 
    'math_place_success', 
    'user_pick_validation', 
    'user_place_validation',
    ]

def main() -> int:
    """main program loop

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
        cfg_filename=CFG_FILE
    )
    data_processor = DataProcessor()

    if DO_PREDICTION:
        data_drawer = DataDrawer(
            admission_pick_rectangle_width=ADMITTED_PICK_WIDTH, 
            admission_pick_rectangle_height=ADMITTED_PICK_HEIGHT,
            admission_place_rectangle_dimension=ADMITTED_PLACE_DIM,
            rows=2,
            cols=2,
            )
    else:
        data_drawer = DataDrawer(
            admission_pick_rectangle_width=ADMITTED_PICK_WIDTH, 
            admission_pick_rectangle_height=ADMITTED_PICK_HEIGHT,
            admission_place_rectangle_dimension=ADMITTED_PLACE_DIM,
            rows=1,
            cols=1,
            )

    all_data_dict = {}

    subdict = {}
    for i in range(len(MODELS)):
        subdict = {}    # empty assignment is necessary or subdict will reference the same variable for all models
        validation_size, training_size, lang_goals = read_model(data_extractor, i, DO_PREDICTION)

        size = 0
        if SET == "both":
            size = training_size + validation_size
        elif SET == "train":
            size = training_size
        elif SET == "val":
            size = validation_size

        user_pick_validation_data = []
        user_place_validation_data = []
        math_pick_success = []
        math_place_success = []

        for index in range(size):
            model_aliases = ALIASES[i]
            episode = data_extractor.get_observation(index, SET)
            (obs, act_actual, _, info) = episode
            
            # replace newer lang goal with alias (old models had different names for objects)
            if info['lang_goal'] in model_aliases:
                info['lang_goal'] = model_aliases[info['lang_goal']]
            
            if DO_PREDICTION:
                # goal is pulled from info if not specified
                act_prediction = data_extractor.act_on_model(obs, info, goal=None)
                subdict[index] = calculate_values(data_extractor, data_processor, data_drawer, act_prediction, episode, size, index, model_aliases)

            data_drawer.set_axs(0, 0)
            data_drawer.draw_im_data(obs['color'], DO_PREDICTION, DO_EVAL_SUCCESS_MATH)
            if DO_PREDICTION:
                hmap = data_processor.augment_hmap(act_prediction['hmap'])
                pick_im = data_processor.augment_confidence_map(act_prediction['pick_confidence'])
                place_im = data_processor.augment_confidence_map(act_prediction['place_confidence'])
                data_drawer.set_axs(0, 1)
                data_drawer.draw_data_to_active_axs(hmap)
                data_drawer.set_axs(1, 0)
                data_drawer.draw_data_to_active_axs(pick_im)
                data_drawer.set_axs(1, 1)
                data_drawer.draw_data_to_active_axs(place_im)
            
            #data_drawer.draw_im_data(obs['color'], DO_PREDICTION, DO_EVAL_SUCCESS_MATH)

            if DO_EVAL_SUCCESS_MATH:
                # pulling class parameters like this is a bad practice
                pick_rect_corners = data_processor.get_angled_rectangle_corners_from_centerpoint(
                    data_drawer.pick_predict[0][0],
                    data_drawer.pick_predict[0][1],
                    ADMITTED_PICK_WIDTH,
                    ADMITTED_PICK_HEIGHT,
                    data_drawer.pick_rot_predict,
                )
                subdict[index][0]['math_pick_success'] = (data_processor.point_inside_polygon(
                    pick_rect_corners, data_drawer.pick_actual[0][:2]
                ))

                place_rect_corners = data_processor.get_angled_rectangle_corners_from_centerpoint(
                    data_drawer.place_predict[0][0],
                    data_drawer.place_predict[0][1],
                    ADMITTED_PICK_WIDTH,
                    ADMITTED_PICK_HEIGHT,
                    data_drawer.place_rot_predict,
                )
                subdict[index][0]['math_place_success'] = (data_processor.point_inside_polygon(
                    place_rect_corners, data_drawer.place_actual[0][:2]
                ))

            if ASK_USER_VALIDATION:
                print(info['lang_goal'])
                commonprompt = f'{MODELS[i] + MODEL_EXTENDERS[i]} example {index+1}/{size}'
                subdict[index][0]['user_pick_validation'] = (input(f'{commonprompt}, pick successful? (y/n)'.lower()) == 'y')
                subdict[index][0]['user_place_validation'] = (input(f'{commonprompt}, place successful? (y/n)'.lower()) == 'y')
                
        
        if DO_EVAL_SUCCESS_MATH:
            subdict['math_pick_success'] = math_pick_success
            subdict['math_place_success'] =  math_place_success
            
        # collect set lang_goals
        subdict["lang_goals"] = lang_goals

        # success validation
        if ASK_USER_VALIDATION:
            subdict["user_pick_validation"] = user_pick_validation_data
            subdict["user_place_validation"] = user_place_validation_data

        if DO_EVAL_SUCCESS_MATH:
            subdict["math_pick_success"] = math_pick_success
            subdict["math_place_success"] = math_place_success

        # stash model data to all data
        all_data_dict[f"{MODELS[i]}{MODEL_EXTENDERS[i]}"] = subdict

        # clear models & agent from memory (for loading new agent)
        data_extractor.clear_model_data()

    if CALCULATE_BOX_VALUES:
        # collapse dict so tasks are not present in multiples but as statistical values
        result_dict, order = data_processor.collapse_results_to_meta_results(
            all_data_dict, USE_BROKEN_GOALS, PLACEHOLDER_VALUE, ERRORS_TO_AVERAGE
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


def read_model(data_extractor: DataHandler, index: int, load_model=True):
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
    
    if load_model:
        data_extractor.find_latest_best_checkpoint_version(
            model, extender
        )
        data_extractor.augment_cfg(model, extender)
        data_extractor.load_model(model, extender)

    return validation_size, training_size, lang_goals


def calculate_values(data_extractor: DataHandler, data_processor: DataProcessor, data_drawer: DataDrawer, act_prediction, episode, size: int, index: int, model_aliases: dict):
    print(f"Task {index+1}/{size}")
    (obs, act_actual, _, info) = episode

    # extract positional values
    pick_actual = act_actual["pose0"]
    place_actual = act_actual["pose1"]
    pick_prediction = act_prediction["pose0"]
    place_prediction = act_prediction["pose1"]

    # extract rotational values
    ert = data_extractor.extract_target_rot
    pick_rot_actual = ert(index, SET, "pick", size)
    place_rot_actual = ert(index, SET, "place", size)

    rom = data_extractor.rot_on_model
    (
        pick_conf,
        pick_rot_pred_conf,
        pick_logits,
    ) = rom(episode, SET, "pick", size)
    (
        place_conf, 
        place_rot_pred_conf, 
        place_logits
    ) = rom(episode, SET, "place")

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
        place_rot_actual,
        info['lang_goal']
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
        csv_text = data_processor.convert_dict_to_csv(all_data_dict[model], order, ERRORS_TO_AVERAGE, DO_BOX_LABELS)
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
