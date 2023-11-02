import hydra
import os
import numpy as np

from omegaconf import DictConfig, OmegaConf
from cliport.model_comparison.comparison_utilities import DataDrawing, DataHandler, DataProcessor


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

def main() -> int:
    data_extractor = DataHandler(root=ROOT_DIR, 
                              exp_path=EXP_FOLDER, 
                              data_path=DATA_FOLDER, 
                              result_path=RESULT_FOLDER)
    data_processor = DataProcessor()
    data_drawer = DataDrawing(mode=MODE, type=TYPE)

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
            pick_conf, pick_rot_pred_conf, pick_logits,  = rom(episode, SET, 'pick')
            place_conf, place_rot_pred_conf, place_logits = rom(episode, SET, 'place')
            
            fp = data_processor.find_rot_peak
            pick_rot_pred = fp(pick_rot_pred_conf)
            place_rot_pred = fp(place_rot_pred_conf)
            
            
            pick_theta = np.unravel_index(np.argmax(pick_rot_pred_conf), shape=pick_rot_pred_conf.shape)[2] * (2 * np.pi / pick_rot_pred_conf.shape[2]) * -1.0
            del ert, rom, fp
            

if __name__ == "__main__":
    os.environ['CLIPORT_ROOT'] = '/home/drubuntu/cliport/cliport'
    main()
