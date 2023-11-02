from cliport.model_comparison.comparison_utilities import DataHandler

ROOT_DIR = '/home/drubuntu/cliport'
RESULT_FOLDER = f'{ROOT_DIR}/comparison_results'
DATA_FOLDER = f'{ROOT_DIR}/data'
EXP_FOLDER = f'{ROOT_DIR}/exps'
data_extractor = DataHandler(root=ROOT_DIR,
                             exp_path=EXP_FOLDER,
                             data_path=DATA_FOLDER,
                             result_path=RESULT_FOLDER)