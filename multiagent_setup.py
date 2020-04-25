# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import gfootball.env as football_env
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np
from arguments import get_args
parser = argparse.ArgumentParser()

parser.add_argument('--num-agents', type=int, default=3)
parser.add_argument('--num-policies', type=int, default=3)
parser.add_argument('--num-iters', type=int, default=100000)
parser.add_argument('--simple', action='store_true')

#class RllibGFootball(MultiAgentEnv):
def create_single_football_env(args):
    env = football_env.create_environment(
        env_name=args.env_name, stacked=True,
        logdir='/tmp/rllib_test',
        write_goal_dumps=False, write_full_episode_dumps=False, render=args.render,
        dump_frequency=0,
        number_of_left_players_agent_controls=args.left_agent,
        number_of_right_players_agent_controls=args.right_agent,
        channel_dimensions=(42, 42))
    return env

class RllibGFootball():
  """An example of a wrapper for GFootball to make it compatible with rllib."""

  def __init__(self, args):
    self.env = create_single_football_env(args)
    self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
    self.observation_space = gym.spaces.Box(
        low=self.env.observation_space.low[0],
        high=self.env.observation_space.high[0],
        dtype=self.env.observation_space.dtype)
    self.num_agents = args.num_agent

  def reset(self):
    original_obs = self.env.reset()
    obs = {}
    for x in range(self.num_agents):
      if self.num_agents > 1:
        obs['agent_%d' % x] = original_obs[x]
      else:
        obs['agent_%d' % x] = original_obs
    return obs

  # def step(self, action):
  #     return self.env.step(action)

  def step(self, action_dict):
    actions = []
    for key, value in sorted(action_dict.items()):
      actions.append(value)
    o, r, d, i = self.env.step(np.array(actions).reshape(self.num_agents,))
    rewards = {}
    obs = {}
    infos = {}
    for pos, key in enumerate(sorted(action_dict.keys())):
      infos[key] = i
      if self.num_agents > 1:
        rewards[key] = r[pos]
        obs[key] = o[pos]
      else:
        rewards[key] = r
        obs[key] = o
    dones = {'__all__': d}
    return obs, rewards, dones, infos

  def close(self):
      self.env.close()


if __name__ == '__main__':
    args = get_args()
    env = RllibGFootball(args)
    # env = create_single_football_env(args)


    obs = env.reset()
    for _ in range(int(1e3)):
        flag = 0
        action = np.random.randint(0, env.action_space.n, 2)
        action_dict = {}
        for i in range(args.num_agent):
            action_dict["agent_{}".format(i)] = action[i]
        obs_next, reward, done, _ = env.step(action_dict)
        if done['__all__']:
            print("done")
            obs_next = env.reset()
        for i in range(args.num_agent):
            if not np.array_equal(obs["agent_{}".format(i)], obs_next["agent_{}".format(i)]):
                print('obs changes')
                print(np.sum(obs["agent_{}".format(i)]-obs_next["agent_{}".format(i)]))
                print(obs["agent_{}".format(i)].shape)
        # if not np.array_equal(obs, obs_next):
        #     flag += 1
        obs = obs_next






