from arguments import get_args
from Selfplay_coach import Coach
import gfootball.env as football_env
from gfootball_net import NNetWrapper as nn
from utils import *
from multiagent_setup import RllibGFootball
import numpy as np
import torch

if __name__ == '__main__':
    is_train = False
    if is_train == True:
        args = get_args()
        env = RllibGFootball(args)
        nnet = nn(env, args)
        c = Coach(env, nnet, args)
        vloss_hist, ploss_hist = c.learn()
        vloss_hist = np.array(vloss_hist)
        ploss_hist = np.array(ploss_hist)
        np.save('ploss_hist.npy', ploss_hist)
        np.save('vloss_hist.npy', vloss_hist)
        env.close()

    if is_train == False:
        model_path = '11_vs_11_competition.pth'
        args = get_args()
        args.render = False # change to True if you want to see the play
        args.left_agent = 5
        args.right_agent = 0
        args.num_agent = 5
        env = RllibGFootball(args)
        network = nn(env, args)
        network.load_state_dict(model_path)
        obs = env.reset()
        for _ in range(int(1e4)):
            action = {}
            for i in range(args.num_agent):
                with torch.no_grad():
                    pi, _ = network.predict(obs['agent_{}'.format(i)])
                action['agent_{}'.format(i)] = np.random.choice(len(pi), p=pi)
            obs, reward, done, _ = env.step(action)
            if done:
                obs = env.reset()
        env.close()
