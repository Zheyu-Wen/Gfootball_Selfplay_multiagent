import math
import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, nnet, args):
        self.env = env
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores value in end of the game

    def getActionProb(self, obs, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        action_dict = {}
        pi_dict = {}
        for i in range(self.args.numMCTSSims):
            self.search(obs)

        for agent_num in range(8):
            counts = [self.Nsa[(str(obs['agent_{}'.format(agent_num)]) + '/' + str(agent_num), a)] if (str(obs['agent_{}'.format(agent_num)]) + '/' + str(agent_num), a)
                                                                          in self.Nsa else 0 for a in range(self.env.action_space.n)]

            if temp == 0:
                bestA = np.argmax(counts)
                probs = [0] * len(counts)
                probs[bestA] = 1
                return probs

            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            counts = [x / counts_sum for x in counts]
            bestA = np.argmax(counts).astype('int')
            action_dict['agent_{}'.format(agent_num)] = bestA
            pi_dict['agent_{}'.format(agent_num)] = counts
        return action_dict, pi_dict

    def search(self, obs):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        value_list = []
        # if done != 0:
        #     for num_agent in range(8):
        #         obs_int = obs['agent_{}'.format(num_agent)]
        #         obs_str = str(obs_int) + '/' + str(num_agent)
        #         if done != 0:
        #             self.Es[obs_str] = -reward
        #         else:
        #             value_list.append(self.Es[obs_str])
        #     return value_list
        for num_agent in range(8):
            obs_int = obs['agent_{}'.format(num_agent)]
            obs_str = str(obs_int) + '/' + str(num_agent)
            # leaf node
            self.Ps[obs_str], v = self.nnet.predict(obs_int)
            sum_Ps_s = np.sum(self.Ps[obs_str])
            self.Ps[obs_str] /= sum_Ps_s  # renormalize
            self.Ns[obs_str] = 0
            value_list.append(-v)

        # valids = self.Vs[s]
        best_act = {}
        for num_agent in range(8):
            cur_best = -float('inf')
            obs_int = obs['agent_{}'.format(num_agent)]
            obs_str = str(obs_int) + '/' + str(num_agent)
            # pick the action with the highest upper confidence bound
            for a in range(self.env.action_space.n):
                # if valids[a]:
                if (obs_str, a) in self.Qsa:
                    u = self.Qsa[(obs_str, a)] + self.args.cpuct * self.Ps[obs_str][a] * math.sqrt(self.Ns[obs_str]) / (
                                1 + self.Nsa[(obs_str, a)])
                else:
                    u = self.args.cpuct * self.Ps[obs_str][a] * math.sqrt(self.Ns[obs_str] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act['agent_{}'.format(num_agent)] = a

        # next_obs, _, _, _ = self.env.step(best_act)
        # v = self.search(next_obs, done, reward)
        for num_agent in range(8):
            obs_int = obs['agent_{}'.format(num_agent)]
            obs_str = str(obs_int) + '/' + str(num_agent)
            if (obs_str, best_act['agent_{}'.format(num_agent)]) in self.Qsa:
                self.Qsa[(obs_str, best_act['agent_{}'.format(num_agent)])] = \
                    (self.Nsa[(obs_str, best_act['agent_{}'.format(num_agent)])] * self.Qsa[(obs_str, best_act['agent_{}'.format(num_agent)])] + value_list[num_agent]) \
                    / (self.Nsa[(obs_str, best_act['agent_{}'.format(num_agent)])] + 1)
                self.Nsa[(obs_str, best_act['agent_{}'.format(num_agent)])] += 1
            else:
                self.Qsa[(obs_str, best_act['agent_{}'.format(num_agent)])] = value_list[num_agent]
                self.Nsa[(obs_str, best_act['agent_{}'.format(num_agent)])] = 1
            self.Ns[obs_str] += 1

        return value_list
