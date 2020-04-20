from collections import deque
# from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, env, nnet, args):
        self.env = env
        self.nnet = nnet
        self.pnet = nnet  # the competitor network
        self.args = args
        self.mcts = MCTS(self.env, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.obs = self.env.reset()
        self.Final_examples = []


    def executeEpisode(self, mcts2=None):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and otherwise
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        if mcts2!=None:
            mcts = mcts2
        else:
            mcts = self.mcts
        trainExamples = []
        episodeStep = 0
        reward = {}
        for num_agent in range(self.args.num_agent):
            reward['agent_{}'.format(num_agent)] = 0
        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            action, pi = mcts.getActionProb(self.obs, reward, temp=temp)
            next_obs, reward, done, _ = self.env.step(action)
            for num_agent in range(self.args.num_agent):
                trainExamples.append([num_agent, self.obs['agent_{}'.format(num_agent)], pi['agent_{}'.format(num_agent)], None, next_obs['agent_{}'.format(num_agent)]])
            self.obs = next_obs

            if episodeStep > 4:
                for x in trainExamples:
                    if x[0] in range(self.args.left_agent):
                        self.Final_examples.append((x[0], x[1], x[2], -1, x[4]))
                    else:
                        self.Final_examples.append((x[0], x[1], x[2], 1, x[4]))
                return self.Final_examples

            elif reward['agent_{}'.format(0)] != 0:
                for x in trainExamples:
                    if x[0] in range(self.args.left_agent):
                        self.Final_examples.append((x[0], x[1], x[2], reward['agent_{}'.format(x[0])], x[4]))
                    else:
                        self.Final_examples.append((x[0], x[1], x[2], -reward['agent_{}'.format(x[0])], x[4]))
                return self.Final_examples

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        vlosss_hist = []
        ploss_hist = []
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.env, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.env, self.pnet, self.args)

            ploss, vloss = self.nnet.train(trainExamples)
            ploss_hist += ploss
            vlosss_hist += vloss
            nmcts = MCTS(self.env, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            example_pmcts = self.executeEpisode(mcts2=pmcts)
            example_nmcts = self.executeEpisode(mcts2=nmcts)
            pwins = 0
            nwins = 0
            for x in example_pmcts:
                if x[0] in range(self.args.left_agent):
                    if x[3] == 1:
                        pwins += 1

            for x in example_nmcts:
                if x[0] in range(self.args.left_agent):
                    if x[3] == 1:
                        nwins += 1

            print('NEW/PREV WINS : %d / %d' % (nwins, pwins))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
        return vlosss_hist, ploss_hist

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
