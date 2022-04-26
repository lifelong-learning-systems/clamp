import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from torch import nn
from torch.nn.functional import mse_loss

from tqdm import tqdm


class IdealLearner(nn.Module):
    def __init__(self, nalgos, ntasks, random_state=0):
        super().__init__()
        # general parameters
        self.ntasks = ntasks
        self.nalgos = nalgos
        self.task_difficulty_scale = 1

        # initialize parameters that can be estimated
        torch.manual_seed(random_state)
        self.task_matrix = nn.Parameter(
            torch.tensor(np.random.random((self.ntasks, self.ntasks)) * 2 - 1, requires_grad=True))
        self.task_difficulty = nn.Parameter(torch.tensor(np.random.random((self.ntasks,)), requires_grad=True))
        self.alg_efficiency = nn.Parameter(torch.tensor(np.random.normal(0.5, 0.02, size=nalgos), requires_grad=True))
        self.alg_memory_horizon = nn.Parameter(torch.tensor(np.random.normal(0.5, 0.1, size=nalgos), requires_grad=True))
        self.alg_experience_boost = nn.Parameter(
            torch.tensor(np.random.normal(0.5, 0.02, size=nalgos), requires_grad=True))

        # things that get filled in during use
        self.result = None
        self.sig = None
        self.target = None
        self.lx = None
        self.losses = None
        self.prediction = None

    def forward(self, lx):
        return self.performance(self.ntasks, self.nalgos, self.task_matrix, lx, self.alg_efficiency,
                                self.alg_memory_horizon, self.alg_experience_boost, self.task_difficulity)

    def performance(self, ntasks, nalgos, task_matrix, lx, algo_efficiency, algo_memory, algo_experience_boost,
                    task_difficulty):
        self.result = torch.tensor(np.zeros((nalgos, ntasks, len(lx) + 1)))
        self.sig = torch.tensor(np.zeros((nalgos, ntasks, len(lx) + 1)))
        for algo in range(nalgos):
            for ind, task in enumerate(lx):
                for task2 in range(task_matrix.shape[0]):
                    prior_exp = torch.sum(self.result[algo, task2, ind:ind + 1]) * algo_memory[algo]
                    prior_perf = torch.sum(self.sig[algo, task, ind:ind + 1])
                    self.result[algo, task2, ind + 1] = (prior_exp + task_matrix[task, task2] * algo_efficiency[algo]
                                                         + task_matrix[task, task2] * prior_perf * algo_experience_boost[algo])

                    difficulty = task_difficulty[task2] * self.task_difficulty_scale
                    self.sig[algo, task2, ind + 1] = self.sigmoid(self.result[algo, task2, ind + 1] / difficulty)

        return self.sig

    @staticmethod
    def sigmoid(x):
        return 2. * torch.exp(x) / (torch.exp(x) + 1.) - 1.

    @staticmethod
    def last_seen(lx, task, current_index):
        result = 0
        for i in range(current_index - 1, 0, -1):
            if lx[i] is not task:
                result += 1
            else:
                return result
        return result

    def optimize(self, lx, target, iters=1000):
        optimizer = torch.optim.Adam(self.parameters(), lr=.01)
        self.target = target
        self.lx = lx
        self.losses = []
        for _ in tqdm(range(iters)):
            optimizer.zero_grad()

            prediction = self(lx)
            self.prediction = prediction
            loss = mse_loss(target, prediction)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.task_matrix.clamp_(-1.0, 1.0)
                self.task_difficulity.clamp_min_(0.0)
                self.alg_efficiency.clamp_min_(0.0)
                self.alg_memory_horizon.clamp_(0.0, 1.0)
                self.alg_experience_boost.clamp_min_(0.0)
            self.losses.append(loss.detach().numpy())
        plt.plot(self.losses)

    def print_algo_results(self, algo_names=None):
        data = {'algorithm': range(self.nalgos),
                'efficiency': self.alg_efficiency.detach().numpy(),
                'retention': self.alg_memory_horizon.detach().numpy(),
                'expertise': self.alg_experience_boost.detach().numpy()}
        if algo_names is not None:
            data['algorithm'] = algo_names
        df = pd.DataFrame(data).to_csv(index=False)
        print(df)

    def print_task_transfer(self):
        print(pd.DataFrame(self.task_matrix.detach().numpy()).to_csv(index=False, header=False))
