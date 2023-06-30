"""Main training script."""

import os
from pathlib import Path

import cv2
import numpy as np
import time
import torch
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset, RealRobotDataset, RealRobotMultiDataset

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def depth_to_heatmap(depth) -> None:
    """Normalize depth image to color heatmap for display"""
    valid = (depth != 0)
    ranged = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) # dmin -> 0.0, dmax -> 1.0
    ranged[ranged < 0] = 0 # saturate
    ranged[ranged > 1] = 1
    output = 1.0 - ranged # 0 -> white, 1 -> black
    output[~valid] = 0 # black out invalid
    output **= 1/2.2 # most picture data is gamma-compressed
    output = cv2.normalize(output, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    
    return output


@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Logger
    # wandb_logger = WandbLogger(name=cfg['tag']) if cfg['train']['log'] else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        filepath=os.path.join(checkpoint_path, 'best'),
        save_top_k=1,
        save_last=True,
    )

    # Trainer
    max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']
    trainer = Trainer(
        gpus=cfg['train']['gpu'],
        fast_dev_run=cfg['debug'],
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        automatic_optimization=False,
        check_val_every_n_epoch=max_epochs // 50,
        resume_from_checkpoint=last_checkpoint,
    )

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
        val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    elif 'single' in dataset_type:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
        val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)
    elif 'real' in dataset_type:
        train_ds = RealRobotDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
        val_ds = RealRobotDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)
    else:
        train_ds = RealRobotMultiDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
        val_ds = RealRobotMultiDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    #item, _ = train_ds[0]
    #item_val, _ = val_ds[0]

    #img_val = item_val['img']
    #img = item['img']
    #p0_val = item_val['p0']
    #p0_theta_val = item_val['p0_theta']
    #line_len = 30
    #pick0 = (int(p0_val[0] + line_len/2.0 * np.sin(p0_theta_val)), int(p0_val[1] + line_len/2.0 * np.cos(p0_theta_val)))
    #pick1 = (int(p0_val[0] - line_len/2.0 * np.sin(p0_theta_val)), int(p0_val[1] - line_len/2.0 * np.cos(p0_theta_val)))
    #dimg = img_val[:,:,:3].astype(np.uint8)
    #sp = (pick0[1], pick0[0])
    #ep = (pick1[1], pick1[0])
    #print("Language Goal: ", item_val['lang_goal'])
    #dimg = cv2.line(dimg, sp, ep, (0, 255, 0), 2) 
    #dimg = cv2.circle(dimg, (p0_val[1], p0_val[0]), 2, (0, 0, 255), 2)
    #dimg = cv2.circle(dimg, (pick1[1], pick1[0]), 2, (255, 0, 0), 2)
    #dimg = cv2.circle(dimg, (pick0[1], pick0[0]), 2, (255, 0, 255), 2)


    #rgb = img[:,:,:3] / 255
    #rgb_val = img_val[:,:,:3] / 255
    #depth = img[:,:,3]
    #depth_val = img_val[:,:,3]
    #depth_val_viz = depth_to_heatmap(depth_val)
    #depth_viz = depth_to_heatmap(depth)
    #rgb_mean = np.mean(rgb, axis=(0, 1))
    #rgb_val_mean = np.mean(rgb_val, axis=(0, 1))
    #depth_mean = np.mean(depth)
    #depth_val_mean = np.mean(depth_val)
    #print("RGB Mean: ", rgb_mean)
    #print("Depth Mean: ", depth_mean)
    #print("RGB Val Mean: ", rgb_val_mean)
    #print("Depth Val Mean: ", depth_val_mean)
    #cv2.imshow("img_val", dimg)
    # cv2.imshow("img", img[:,:,:3].astype(np.uint8))
    # cv2.imshow("depth", depth_viz)
    #cv2.imshow("depth_val", depth_val_viz)
    #cv2.waitKey(0)


    # Initialize agent
    agent = agents.names[agent_type](name, cfg, train_ds, val_ds)

    time.sleep(2)

    # # Main training loop
    trainer.fit(agent)

if __name__ == '__main__':
    main()
