import hydra
import os
from omegaconf import DictConfig, OmegaConf

from cliport.model_comparison.comparison_utilities import DataComparison, DataHandler, DataProcessor


ROOT_DIR = '/home/drubuntu/cliport'
RESULT_FOLDER = f'{ROOT_DIR}/comparison_results'
DATA_FOLDER = f'{ROOT_DIR}/data'
EXP_FOLDER = f'{ROOT_DIR}/exps'
MODELS = ['engine-parts-to-box-single-list',
          'engine-parts-single', 
          'packing-objects',
          'packing-objects']
# extenders are required because of requirement to test the same model with 
# multiple training examples. These extenders could be programmatically found as well.
# (mainly because previous project results are compared to current project results / data pollution)
MODEL_EXTENDERS = ['-cliport-n88-train',
                   '-cliport-n34-train',
                   '-cliport-n108-train',
                   '-cliport-n119-train']
MODE = 'single'
TYPE = 'common'
SET = 'val'

def main() -> None:
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

  data_extractor = DataHandler(root=ROOT_DIR, 
                              exp_path=EXP_FOLDER, 
                              data_path=DATA_FOLDER, 
                              result_path=RESULT_FOLDER)
  data_processor = DataProcessor()
  data_comparer = DataComparison(mode=MODE, type=TYPE)

  all_data_dict = {}

  for i in range(len(MODELS)):
    
    subdict = {}
    
    print(f"\nReading model: {MODELS[i]} ({i+1}/{len(MODELS)})\n")
    # read values from directories (data storage inside extractor)
    training_size = data_extractor.get_set_limit(MODELS[i], 'train')
    validation_size = data_extractor.get_set_limit(MODELS[i], 'val')
    lang_goals = data_extractor.get_lang_goals(MODELS[i])
    data_extractor.read_dataset(MODELS[i], 'train')
    data_extractor.read_dataset(MODELS[i], 'val')
    data_extractor.find_latest_best_checkpoint_version(MODELS[i], MODEL_EXTENDERS[i])
    data_extractor.augment_cfg(MODELS[i], MODEL_EXTENDERS[i])
    data_extractor.load_model(MODELS[i], MODEL_EXTENDERS[i])
    
    size = training_size if SET == 'train' else validation_size
    for index in range(size):
      print(f"Example {index+1}/{size}")
      episode = data_extractor.get_observation(index, SET)
      (obs, act_actual, _, info) = episode
      # goal is pulled from info if not specified
      act_prediction = data_extractor.act_on_model(obs, info, goal=None)
      
      # extract positional values
      pick_actual = act_actual['pose0']
      place_actual = act_actual['pose1']
      pick_prediction = act_prediction['pose0']
      place_prediction = act_prediction['pose1']
      
      # extract rotational values
      ert = data_extractor.extract_target_rot
      pick_rot_actual = ert(index, SET, 'pick')
      place_rot_actual = ert(index, SET, 'place')
      
      rom = data_extractor.rot_on_model
      pick_rot_pred_conf, pick_logits,  = rom(episode, SET, 'pick')
      place_rot_pred_conf, place_logits = rom(episode, SET, 'place')
      
      fp = data_processor.find_peak
      pick_rot_pred = fp(pick_rot_pred_conf)
      place_rot_pred = fp(place_rot_pred_conf)
      
      # calculate errors
      cpd = data_processor.calculate_pythagorean_distance
      pick_dist_err_collection = cpd(pick_actual, pick_prediction)
      place_dist_err_collection = cpd(place_actual, place_prediction)
      
      cad = data_processor.calculate_angular_distance
      pick_rot_err = cad(pick_rot_actual, pick_rot_pred)
      place_rot_err = cad(place_rot_actual, place_rot_pred)
      
      travel_errors = data_processor.calculate_travel_error(pick_actual, 
                                                            place_actual, 
                                                            pick_prediction, 
                                                            place_prediction)
      
      del ert, rom, fp, cpd, cad
      
      subdict[index] = [
        {
          'pick_x_error': pick_dist_err_collection[0],
          'pick_y_error': pick_dist_err_collection[1],
          'pick_z_error': pick_dist_err_collection[2],
          'total_pick_error': pick_dist_err_collection[3],
          'place_x_error': place_dist_err_collection[0],
          'place_y_error':place_dist_err_collection[1],
          'place_z_error':place_dist_err_collection[2],
          'total_place_error': place_dist_err_collection[3],
          'pick_rotation_error': pick_rot_err,
          'place_rotation_error': place_rot_err,
          'travel_error': travel_errors[0],
          'actual_travel': travel_errors[1],
          'predicted_travel': travel_errors[2],
        },
        info['lang_goal']
      ]
      
    
    subdict['lang_goals'] = lang_goals
    all_data_dict[f'{MODELS[i]}{MODEL_EXTENDERS[i]}'] = subdict
    
    # clear models & agent from memory
    data_extractor.clear_model_data()
    
  # collapse dict so tasks are not present in multiples but as statistical values 
  all_data_dict = data_processor.collapse_results_to_meta_results(all_data_dict)
  for model in all_data_dict:
    csv_text = data_processor.convert_dict_to_csv(all_data_dict[model])
    data_extractor.write_csv_to_disk(csv_text, f"{model}.csv")

  # Write all data to a csv for future processing
  
  # free up rogue memory (?)
  del data_extractor
  print("Data extraction complete")
  return(0)

if __name__ == "__main__":
    os.environ['CLIPORT_ROOT'] = '/home/drubuntu/cliport/cliport'
    main()
