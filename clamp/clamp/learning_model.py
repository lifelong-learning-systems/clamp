import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from torch import nn

from tqdm import tqdm


class IdealLearner(nn.Module):
    def __init__(self, nalgos, ntasks):
        super().__init__()
        # initialize weights with random numbers
        self.ntasks = ntasks
        self.nalgos = nalgos
        self.min_perf = 0.
        self.max_perf = 100
        self.asymptote = 1
        self.task_difficulty_scale = 1
        self.memory_horizon_scale = 1
        self.task_matrix = nn.Parameter(
            torch.tensor(np.random.random((self.ntasks, self.ntasks)) * 2 - 1, requires_grad=True))
        self.task_difficulity = nn.Parameter(torch.tensor(np.random.random((self.ntasks)), requires_grad=True))
        self.alg_efficiency = nn.Parameter(torch.tensor(np.random.normal(0.5, 0.02, size=nalgos), requires_grad=True))
        self.alg_memory_horizon = nn.Parameter(torch.tensor(np.random.normal(.5, .1, size=nalgos), requires_grad=True))
        self.alg_experience_boost = nn.Parameter(
            torch.tensor(np.random.normal(0.5, 0.02, size=nalgos), requires_grad=True))

    def forward(self, lx):
        return self.performance(self.ntasks, self.nalgos, self.task_matrix, lx, self.alg_efficiency,
                                self.alg_memory_horizon, self.alg_experience_boost, self.task_difficulity)

    def performance(self, ntasks, nalgos, task_matrix, lx, algo_efficiency, algo_memory, algo_experience_boost,
                    task_difficulity):
        self.result = torch.tensor(np.zeros((nalgos, ntasks, len(lx) + 1)))
        self.sig = torch.tensor(np.zeros((nalgos, ntasks, len(lx) + 1)))
        for algo in range(nalgos):
            for ind, task in enumerate(lx):
                for task2 in range(task_matrix.shape[0]):
                    prior_exp = torch.sum(self.result[algo, task2, ind:ind + 1]) * algo_memory[algo]
                    prior_perf = torch.sum(self.sig[algo, task, ind:ind + 1])
                    self.result[algo, task2, ind + 1] = prior_exp + task_matrix[task, task2] * algo_efficiency[algo] + \
                                                        task_matrix[task, task2] * prior_perf * algo_experience_boost[
                                                            algo]

                    difficulty = task_difficulity[task2] * self.task_difficulty_scale
                    self.sig[algo, task2, ind + 1] = (1 / (
                                1 + (torch.exp(-1 * self.result[algo, task2, ind + 1] / difficulty))) * 2) - 1

        return self.sig

    def sigmoid(self, x):
        return 2. * torch.exp(x) / (torch.exp(x) + 1.) - 1.

    def last_seen(self, lx, task, current_index):
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
        for i in tqdm(range(iters)):
            optimizer.zero_grad()

            prediction = self(lx)
            self.prediction = prediction
            # pdb.set_trace()
            loss = torch.nn.functional.mse_loss(target, prediction)
            loss.backward()
            with torch.no_grad():
                self.task_matrix.clamp_(-1, 1)
                self.task_difficulity.clamp_min_(0)
                self.alg_efficiency.clamp_min_(0)
                self.alg_memory_horizon.clamp_(0, 1)
                self.alg_experience_boost.clamp_min_(0)
            optimizer.step()
            self.losses.append(loss.detach().numpy())
        plt.plot(self.losses)

    def print_algo_results(self, algo_names=None):
        data = {'algorithm': range(self.nalgos),
                'efficiency': self.alg_efficiency.detach().numpy(),
                'retention': self.alg_memory_horizon.detach().numpy(),
                'expertice': self.alg_experience_boost.detach().numpy()}
        if algo_names is not None:
            data['algorithm'] = algo_names
        df = pd.DataFrame(data).to_csv(index=False)
        print(df)

    def print_task_transfer(self):
        print(pd.DataFrame(self.task_matrix.detach().numpy()).to_csv(index=False, header=False))
