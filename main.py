from arguments import get_args
from Selfplay_coach import Coach
import gfootball.env as football_env
from gfootball_net import NNetWrapper as nn
from utils import *
from multiagent_setup import RllibGFootball
import numpy as np

if __name__ == '__main__':
    args = get_args()
    env = RllibGFootball(args)
    nnet = nn(env)
    c = Coach(env, nnet, args)
    vloss_hist, ploss_hist = c.learn()
    vloss_hist = np.array(vloss_hist)
    ploss_hist = np.array(ploss_hist)
    np.save('ploss_hist.npy', ploss_hist)
    np.save('vloss_hist.npy', vloss_hist)
    env.close()