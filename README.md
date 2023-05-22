# CLIPort

[**CLIPort: What and Where Pathways for Robotic Manipulation**](https://arxiv.org/pdf/2109.12098.pdf)  
[Mohit Shridhar](https://mohitshridhar.com/), [Lucas Manuelli](http://lucasmanuelli.com/), [Dieter Fox](https://homes.cs.washington.edu/~fox/)  
[CoRL 2021](https://www.robot-learning.org/) 

CLIPort is an end-to-end imitation-learning agent that can learn a single language-conditioned policy for various tabletop tasks. The framework combines the broad semantic understanding (_what_) of [CLIP](https://openai.com/blog/clip/) with the spatial precision (_where_) of [TransporterNets](https://transporternets.github.io/) to learn generalizable skills from limited training demonstrations.

For the latest updates, see: [cliport.github.io](https://cliport.github.io)

![](media/sim_tasks.gif)

## Guides

- Getting Started: [Installation](#installation), [Quick Tutorial](#quickstart), [Checkpoints & Objects](#download), [Hardware Requirements](#hardware-requirements), [Model Card](model-card.md)
- Data Generation: [Dataset](#dataset-generation), [Tasks](cliport/tasks)
- Training & Evaluation: [Single Task](#single-task-training--evaluation), [Multi Task](#multi-task-training--evaluation)
- Miscellaneous: [Notebooks](#notebooks), [Docker Guide](#docker-guide), [Disclaimers](#disclaimers--limitations), [Real-Robot Training FAQ](#real-robot-training-faq), [Recording Videos](#recording-videos)
- References: [Citations](#citations), [Acknowledgements](#acknowledgements)

## Installation

Clone Repo:
```bash
git clone https://github.com/cliport/cliport.git
```

Setup virtualenv and install requirements:
```bash
# setup virtualenv with whichever package manager you prefer
virtualenv -p $(which python3.8) --system-site-packages cliport_env  
source cliport_env/bin/activate
pip install --upgrade pip

cd cliport
pip install -r requirements.txt

export CLIPORT_ROOT=$(pwd)
python setup.py develop
```

**Note**: You might need versions of `torch==1.7.1` and `torchvision==0.8.2` that are compatible with your CUDA and hardware. 

## Quickstart

A quick tutorial on evaluating a pre-trained multi-task model.

Download a [pre-trained checkpoint](https://github.com/cliport/cliport/releases/download/v1.0.0/cliport_quickstart.zip) for `multi-language-conditioned` trained with 1000 demos:
```bash
sh scripts/quickstart_download.sh
```

Generate a small `test` set of 10 instances for `stack-block-pyramid-seq-seen-colors` inside  `$CLIPORT_ROOT/data`:
```bash
python cliport/demos.py n=10 \
                        task=stack-block-pyramid-seq-seen-colors \
                        mode=test 
```   
This will take a few minutes to finish. 

Evaluate the best validation checkpoint for `stack-block-pyramid-seq-seen-colors` on the test set:
```bash
python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=10 \
                       train_demos=1000 \
                       exp_folder=cliport_quickstart \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=True
```
If you are on a headless machine turn off the visualization with `disp=False`. 

You can evaluate the same `multi-language-conditioned` model on other tasks. First generate a `val` set for the task and then specify `eval_task=<task_name>` with `mode=val` and `checkpoint_type=val_missing` (the quickstart doesn't include validation results for all tasks; download all task results from [here](#download)).

Checkout [affordance.ipynb](notebooks/affordances.ipynb) to visualize affordance predictions of `cliport` on various tasks. 

## Download

### Google Scanned Objects

Download center-of-mass (COM) corrected [Google Scanned Objects](https://github.com/cliport/cliport/releases/download/v1.0.0/google.zip):
```bash
sh scripts/google_objects_download.sh
```
Credit: [Google](#acknowledgements).

### Pre-trained Checkpoints and Result JSONs
This [Google Drive Folder](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfk1TZjhhbXN2dDFSSzktdU5UN3MtbGxfYXNBWlR2SDN0OUdzMkVrdnU3TU0?resourcekey=0-upqOBPNOlOrAzI7FnQuCiQ&usp=share_link) contains pre-trained `multi-language-conditioned` checkpoints for `n=1,10,100,1000` and validation/test result JSONs for all tasks. The `*val-results.json` files contain the name of the best checkpoint (from validation) to be evaluated on the `test` set.

**Note:** Google Drive might complain about bandwidth restrictions. I recommend using [rclone](https://rclone.org/drive/) with API access enabled.

Evaluate the best validation checkpoint on the test set:
```bash
python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=10 \
                       train_demos=100 \
                       exp_folder=cliport_exps \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=True
```

## Hardware Requirements 

A single NVIDIA GPU with **8.5** to **9.5GB** memory should be sufficient for training and evaluation.

Tested with:
- **GPU** - NVIDIA P100
- **CPU** - Intel Xeon (Quad Core)
- **RAM** - 32GB
- **OS** - Ubuntu 16.04, 18.04

## Training and Evaluation

The following is a guide for training everything from scratch. All tasks follow a 4-phase workflow:
 
1. Generate `train`, `val`, `test` datasets with `demos.py` 
2. Train agents with `train.py` 
3. Run validation with `eval.py` to find the best checkpoint on `val` tasks and save `*val-results.json`
4. Evaluate the best checkpoint in `*val-results.json` on `test` tasks with `eval.py`

### Dataset Generation

#### Single Task

Generate a `train` set of 1000 demonstrations for `stack-block-pyramid-seq-seen-colors` inside `$CLIPORT_ROOT/data`:
```bash
python cliport/demos.py n=1000 \
                        task=stack-block-pyramid-seq-seen-colors \
                        mode=train 
```

You can also do a sequential sweep with `-m` and comma-separated params `task=towers-of-hanoi-seq-seen-colors,stack-block-pyramid-seq-seen-colors`. Use `disp=True` to visualize the data generation.

#### Full Dataset

Run [`generate_dataset.sh`](scripts/generate_datasets.sh) to generate the full dataset and save it to `$CLIPORT_ROOT/data`:

```bash
sh scripts/generate_dataset.sh data
```
**Note:** This script is not parallelized and will take a long time (maybe days) to finish. The full dataset requires [~1.6TB of storage](https://i.kym-cdn.com/photos/images/newsfeed/000/515/629/9bd.gif), which includes both language-conditioned and demo-conditioned (original TransporterNets) tasks. It's recommend that you start with single-task training if you don't have enough storage space.

### Single-Task Training & Evaluation

Make sure you have a `train` (n demos) and `val` (100 demos) set for the task you want to train on.

#### Training

Train a `cliport` agent with `1000` demonstrations on the `stack-block-pyramid-seq-seen-colors` task for 200K iterations:

```bash
python cliport/train.py train.task=stack-block-pyramid-seq-seen-colors \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=1000 \
                        train.n_steps=201000 \
                        train.exp_folder=exps \
                        dataset.cache=False 
```

#### Validation

Iteratively evaluate all the checkpoints on `val` and save the results in `exps/<task>-train/checkpoints/<task>-val-results.json`: 

```bash
python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       exp_folder=exps 
```

#### Test

Choose the best checkpoint from validation to run on the `test` set and save the results in `exps/<task>-train/checkpoints/<task>-test-results.json`:

```bash
python cliport/eval.py eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       exp_folder=exps 
```

### Multi-Task Training & Evaluation

#### Training

Train multi-task models by specifying `task=multi-language-conditioned`, `task=multi-attr-packing-box-pairs-unseen-colors` etc.

```bash
python cliport/train.py train.task=multi-language-conditioned \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=1000 \
                        train.n_steps=601000 \
                        dataset.cache=False \
                        train.exp_folder=exps \
                        dataset.type=multi 
```

**Important**: You need to generate the full dataset of tasks specified in [`dataset.py`](cliport/dataset.py) before multi-task training or modify the list of tasks [here](cliport/dataset.py#L392). 

#### Validation

Run validation with a trained `multi-language-conditioned` multi-task model on `stack-block-pyramid-seq-seen-colors`:

```bash
python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       type=single \
                       exp_folder=exps 
```

#### Test

Evaluate the best checkpoint on the `test` set:

```bash
python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       type=single \
                       exp_folder=exps 
```

## Recording Videos

To save high-resolution videos of agent executions, set `record.save_video=True`:

```bash
python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=10 \
                       train_demos=100 \
                       exp_folder=cliport_exps \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=True \
                       record.save_video=True
```

This will save videos inside `${model_dir}/${exp_folder}/${eval_task}-${agent}-n${train_demos}-train/videos/`.  

**Note:** Rendering at high-resolutions is super slow and will take a long time to finish.

## Disclaimers & Limitations

- **Code Quality Level**: Tired grad student. 
- **Scaling**: The code only works for batch size 1. See [#issue1](https://github.com/cliport/cliport/issues/1) for reference. In theory, there is nothing preventing larger batch sizes other than GPU memory constraints.
- **Memory and Storage**: There are lots of places where memory usage can be reduced. You don't need 3 copies of the same CLIP ResNet50 and you don't need to save its weights in checkpoints since it's frozen anyway. Dataset sizes could be dramatically reduced with better storage formats and compression. 
- **Frameworks**: There are lots of leftover NumPy bits from when I was trying to reproduce the TransportNets results. I'll try to clean up when I get some time. 
- **Rotation Augmentation**: All tasks use the same distribution for sampling SE(2) rotation perturbations. This obviously leads to issues with tasks that involve spatial relationships like 'left' or 'forward'.  
- **Evaluation Runs**: In an ideal setting, the evaluation metrics should be averaged over 3 or more repetitions with different seeds. This might be feasible if you are working just with multi-task models. 
- **Duplicate Training Sets**: The train sets of some `*seen` and `*unseen` tasks are identical, and only the val and test sets differ for purposes of evaluating generalization performance. So you might not need two duplicate train sets or train two separate models.   
- **Image Resolution**: The input resolution of `320 x 160` might be too small for some tasks with tiny objects, especially for packing Google objects. Larger resolutions might help improve legibility.   
- **Disadvantaged Multi-Task Models**: To avoid cheating on `packing-seen-google-object-*` tasks, the multi-task models are never trained on the full `seen` split of Google Scanned Objects. So a single-task model trained on `packing-seen-google-object-*` will have seen more objects than the comparable multi-task model.
- **Other Limitations**: Checkout Appendix I in the paper.

## Notebooks

- [CLIP Playground in Colab](https://github.com/kevinzakka/clip_playground) by [Kevin Zakka](https://kzakka.com/): A zero-shot object detector with just CLIP. Note that CLIPort does not 'detect objects' but instead directly 'detects actions'.
- [Dataset Visualizer](notebooks/dataset.ipynb): Visualizes raw data and expert labels for pre-generated datasets. 
- [Affordance Heatmaps](notebooks/affordances.ipynb): Visualizes affordances from a pre-trained agent like `cliport`.
- [Evaluation Results](notebooks/results.ipynb): Prints success scores from evaluation runs.

## Docker Guide

Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker#ubuntu-160418042004-debian-jessiestretchbuster). 

Modify [docker_build.py](scripts/docker_build.py) and [docker_run.py](scripts/docker_run.py) to your needs.

#### Build 

Build the image:

```bash
python scripts/docker_build.py 
```

#### Run

Start container:

```bash
python scripts/docker_run.py --nvidia_docker
 
  cd ~/cliport
```

Use `scripts/docker_run.py --headless` if you are on a headless machines like a remote server or cloud instance.

## Real-Robot Training FAQ

#### How much training data do I need?

It depends on the complexity of the task. With 5-10 demonstrations the agent should start to do something useful, but it will often make mistakes by picking the wrong object. For robustness you probably need 50-100 demostrations. A good way to gauge how much data you might need is to setup a simulated version of the problem and evaluate agents trained with 1, 10, 100, and 1000 demonstrations.  

#### Why doesn't the agent follow my language instruction?

This means either there is some sort of bias in the dataset that the agent is exploiting, or you don't have enough training data. Also make sure that the task is doable - if a referred attribute is barely legible in the input, then it's going to be hard for agent to figure out what you mean. 

#### Does CLIPort predict height (z-values) of the end-effector? #### 

CLIPort does not predict height values. You can either: (1) come up with a heuristic based on the heightmap to determine the height position, or (2) train a simple MLP like in [TransportNets-6DOF](https://github.com/google-research/ravens/blob/master/ravens/models/transport_6dof.py) to predict z-values.

#### Shouldn't CLIP help in zero-shot detection of things? Why do I need collect more data?

Note that CLIPort is not doing "object detection". CLIPort fine-tunes CLIP's representations to "detect actions" in SE(2). CLIP by itself has no understanding of actions or affordances; recognizing and localizing objects (e.g. detecting hammer) does not tell you anything about how to manipulate them (e.g. grasping the hammer by the handle).    

#### What are the best hyperparams for real-robot training?

The [default settings](cliport/cfg/train.yaml) should work well. Although recently, I have been playing around with using FiLM [(Perez et. al, 2017)](https://distill.pub/2018/feature-wise-transformations/) to fuse language features inspired by BC-0 [(Jang et. al, 2021)](https://openreview.net/forum?id=8kbp23tSGYv). Qualitatively, it seems like FiLM is better for reading text etc. but I haven't conducted a full quantitative analysis. Try it out yourself with `train.agent=two_stream_clip_film_lingunet_lat_transporter` (non-residual FiLM).      

#### How to pick the best checkpoint for real-robot tasks?

Ideally, you should create a validation set with heldout instances and then choose the checkpoint with the lowest translation and rotation errors. You can also reuse the training instances but swap the language instructions with unseen goals.

#### Why is the agent confusing directions like 'forward' and 'left'?

By default, training samples are augmented with SE(2) rotations sampled from `N(0, 60 deg)`. For tasks with rotational symmetries (like moving pieces on a chessboard) you need to be careful with this [rotation augmentation parameter](cliport/cfg/train.yaml#L15).


## Acknowledgements

This work use code from the following open-source projects and datasets:

#### Google Ravens (TransporterNets)
Original:  [https://github.com/google-research/ravens](https://github.com/google-research/ravens)  
License: [Apache 2.0](https://github.com/google-research/ravens/blob/master/LICENSE)    
Changes: All PyBullet tasks are directly adapted from the Ravens codebase. The original TransporterNets models were reimplemented in PyTorch.

#### OpenAI CLIP

Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  
Changes: Minor modifications to CLIP-ResNet50 to save intermediate features for skip connections.

#### Google Scanned Objects

Original: [Dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)  
License: [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Changes: Fixed center-of-mass (COM) to be geometric-center for selected objects.

#### U-Net 

Original: [https://github.com/milesial/Pytorch-UNet/](https://github.com/milesial/Pytorch-UNet/)  
License: [GPL 3.0](https://github.com/milesial/Pytorch-UNet/)  
Changes: Used as is in [unet.py](cliport/models/core/unet.py). Note: This part of the code is GPL 3.0.  

## Citations

**CLIPort**
```bibtex
@inproceedings{shridhar2021cliport,
  title     = {CLIPort: What and Where Pathways for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL)},
  year      = {2021},
}
```

**CLIP**
```bibtex
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

**TransporterNets**
```bibtex
@inproceedings{zeng2020transporter,
  title={Transporter networks: Rearranging the visual world for robotic manipulation},
  author={Zeng, Andy and Florence, Pete and Tompson, Jonathan and Welker, Stefan and Chien, Jonathan and Attarian, Maria and Armstrong, Travis and Krasin, Ivan and Duong, Dan and Sindhwani, Vikas and others},
  booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
  year= {2020},
}
```

## Questions or Issues?

Please file an issue with the issue tracker.  
