from arguments import get_args
from Selfplay_coach import Coach
import gfootball.env as football_env
from gfootball_net import NNetWrapper as nn
from utils import *
from multiagent_setup import RllibGFootball


if __name__ == '__main__':
    args = get_args()
    num_agents = 8
    env = RllibGFootball(num_agents)
    nnet = nn(env)
    c = Coach(env, nnet, args)
    c.learn()