import os
import sys
import json
import time

import numpy as np
from cliport import tasks
from cliport import agents
from cliport.utils import utils

import torch
import cv2
from cliport.dataset import RealRobotDataset

import numpy as np


n_eval = 1 # number of evaluation instances
# mode = 'test' # val or test

agent_type = 'two_stream_clip_lingunet_lat_transporter'

model_folder = 'exps/packing-objects-cliport-n108-train/checkpoints' # path to pre-trained checkpoint
ckpt_name = 'best.ckpt' # name of checkpoint to load

### Uncomment the task you want to evaluate on ###
eval_task = 'packing-objects'

root_dir = os.environ['CLIPORT_ROOT']
config_file = 'train.yaml' 

cfg = utils.load_hydra_config(os.path.join(root_dir, f'cliport/cfg/{config_file}'))
data_dir = os.path.join(root_dir, "data/packing-objects-train")

# vcfg['mode'] = mode
# vcfg['model_task'] = model_task
# vcfg['eval_task'] = eval_task
# vcfg['agent'] = agent_name

# Load dataset
ds = RealRobotDataset(data_dir, cfg, n_demos=50, augment=False)

eval_run = 0
name = '{}-{}-{}-{}'.format(eval_task, agent_type, n_eval, eval_run)
print(f'\nEval ID: {name}\n')

# Initialize agent
utils.set_seed(eval_run, torch=True)
agent = agents.names[agent_type](name, cfg, None, ds)

# Load checkpoint
ckpt_path = os.path.join(root_dir, model_folder, ckpt_name)
print(f'\nLoading checkpoint: {ckpt_path}')
agent.load(ckpt_path)

episode, _ = ds.load(0, True, False)

# Return random observation action pair from episode.
episode = episode[0]
(obs, act, _, info) = episode

# Get action predictions
tt = time.time()
act = agent.act(obs, info, goal=None)
print("Time taken on inference: ", time.time() - tt)
pick, place = act['pick'], act['place']
