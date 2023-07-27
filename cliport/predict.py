import os
import sys
import json

import numpy as np
from cliport import tasks
from cliport import agents
from cliport.utils import utils

import torch
import cv2
from cliport.dataset import RealRobotDataset

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def main_loop():
    train_demos = 1000  # number training demonstrations used to train agent
    n_eval = 1  # number of evaluation instances
    # mode = 'test' # val or test

    agent_type = 'two_stream_clip_lingunet_lat_transporter'

    model_folder = 'exps/engine-parts-to-box-single-list-cliport-n88-train/checkpoints'  # path to trained checkpoint
    ckpt_name = 'best-v2.ckpt'  # name of checkpoint to load

    draw_grasp_lines = True
    affordance_heatmap_scale = 30

    # Uncomment the task you want to evaluate on
    eval_task = 'packing-objects'

    root_dir = os.environ['CLIPORT_ROOT']
    config_file = 'train.yaml'

    cfg = utils.load_hydra_config(os.path.join(root_dir, f'cliport/cfg/{config_file}'))
    data_dir = os.path.join(root_dir, "data/engine-parts-to-box-single-list-train")

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

    for episode_id in range(len(ds.sample_set)):
        fig, axs = plt.subplots(2, 2, figsize=(13, 7))
        # Get batch
        # episode_id = np.random.choice(ds.sample_set)
        episode, _ = ds.load(episode_id, True, False)

        # Return random observation action pair from episode.
        episode = episode[0]
        (obs, act, _, info) = episode

        # Process sample.
        batch = ds.process_sample(episode, augment=False)

        fig.suptitle(batch['lang_goal'], fontsize=16)

        # Get color and depth inputs
        img = batch['img']
        img = torch.from_numpy(img)
        color = np.uint8(img.detach().cpu().numpy())[:, :, :3]
        color = color.transpose(1, 0, 2)
        depth = np.array(img.detach().cpu().numpy())[:, :, 3]
        depth = depth.transpose(1, 0)

        # Display input color
        axs[0, 0].imshow(color)
        axs[0, 0].axes.xaxis.set_visible(False)
        axs[0, 0].axes.yaxis.set_visible(False)
        axs[0, 0].set_title('Input RGB')

        # Display input depth
        axs[0, 1].imshow(depth)
        axs[0, 1].axes.xaxis.set_visible(False)
        axs[0, 1].axes.yaxis.set_visible(False)
        axs[0, 1].set_title('Input Depth')

        # Display predicted pick affordance
        axs[1, 0].imshow(color)
        axs[1, 0].axes.xaxis.set_visible(False)
        axs[1, 0].axes.yaxis.set_visible(False)
        axs[1, 0].set_title('Pick Affordance')

        # Display predicted place affordance
        axs[1, 1].imshow(color)
        axs[1, 1].axes.xaxis.set_visible(False)
        axs[1, 1].axes.yaxis.set_visible(False)
        axs[1, 1].set_title('Place Affordance')

        # Get action predictions
        l = str(batch['lang_goal'])
        act = agent.act(obs, info, goal=None)
        pick, place = act['pick'], act['place']

        # Visualize pick affordance
        pick_inp = {'inp_img': batch['img'], 'lang_goal': l}
        pick_conf = agent.attn_forward(pick_inp)
        logits = pick_conf.detach().cpu().numpy()
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0 = argmax[:2]

        pick_rot_inp = {'inp_img': batch['img'], 'lang_goal': l, 'p0': p0, }
        pick_rot_conf = agent.attn_rot_forward(pick_rot_inp)
        pick_rot_conf = pick_rot_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_rot_conf)
        argmax = np.unravel_index(argmax, shape=pick_rot_conf.shape)
        p0_theta = argmax[2] * (2 * np.pi / pick_rot_conf.shape[2]) * -1.0

        line_len = 30
        pick0 = (pick[0] + line_len / 2.0 * np.sin(p0_theta), pick[1] + line_len / 2.0 * np.cos(p0_theta))
        pick1 = (pick[0] - line_len / 2.0 * np.sin(p0_theta), pick[1] - line_len / 2.0 * np.cos(p0_theta))

        if draw_grasp_lines:
            axs[1, 0].plot((pick1[0], pick0[0]), (pick1[1], pick0[1]), color='r', linewidth=1)

        # Visualize place affordance
        place_inp = {'inp_img': batch['img'], 'p0': pick, 'lang_goal': l}
        place_conf = agent.trans_forward(place_inp)

        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = (argmax[2] * (2 * np.pi / place_conf.shape[2]) + p0_theta) * -1.0

        line_len = 30
        place0 = (place[0] + line_len / 2.0 * np.sin(p1_theta), place[1] + line_len / 2.0 * np.cos(p1_theta))
        place1 = (place[0] - line_len / 2.0 * np.sin(p1_theta), place[1] - line_len / 2.0 * np.cos(p1_theta))

        if draw_grasp_lines:
            axs[1, 1].plot((place1[0], place0[0]), (place1[1], place0[1]), color='g', linewidth=1)

        # Overlay affordances on RGB input
        pick_logits_disp = np.uint8(logits * 255 * affordance_heatmap_scale).transpose(1, 0, 2)
        place_logits_disp = np.uint8(np.sum(place_conf, axis=2)[:, :, None] * 255 * affordance_heatmap_scale).transpose(
            1, 0, 2)

        pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)
        place_logits_disp_masked = np.ma.masked_where(place_logits_disp < 0, place_logits_disp)

        axs[1][0].imshow(pick_logits_disp_masked, alpha=0.75)
        axs[1][1].imshow(place_logits_disp_masked, cmap='viridis', alpha=0.75)

        # plt.cla()

        # print(f"Lang Goal: {str(batch['lang_goal'])}")
        print(f"Saving figure {episode_id} ...")
        plt.savefig(os.path.join(root_dir, f"figures/train-latest/{episode_id}.png"))
        # plt.show()


if __name__ == "__main__":
    main_loop()
