#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import utils
import hydra

from logger import Logger
from replay_buffer import ReplayBuffer

import matplotlib
from matplotlib import pyplot as plt
import time

import cv2

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)

        self.device = torch.device(cfg.device)
        self.log_success = False
        self.step = 0
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        
        # no relabel
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))


    def evaluate(self):
        
        cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE) 

        average_episode_reward = 0
        if self.log_success:
            success_rate = 0
            
        for episode in range(self.cfg.num_eval_episodes):
            print("%d episode"%episode)
            obs = self.env.reset()
            self.agent.reset()

            done = False
            episode_reward = 0
            
            if self.log_success:
                episode_success = 0

            count = 0
            while not done:

                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                rgb_array = self.env.render('rgb_array')/255.0

                rgb_array = np.float32(rgb_array)
                img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                print(img.size)
                cv2.imshow("output",img)
                time.sleep(0.02)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

                episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            

            average_episode_reward += episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
    
    def load_agent(self, model_dir, step):
        self.agent.load(model_dir, step)
        
@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.load_agent("/home/jackyoung96/MAPbRL/BPref/exp/walker_walk/H1024_L2_B1024_tau0.005/sac_unsup0_topk5_sac_lr0.0005_temp0.1_seed12345", 1000000)
    workspace.evaluate()

if __name__ == '__main__':
    main()
