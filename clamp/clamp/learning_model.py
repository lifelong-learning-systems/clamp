import numpy as np
import pandas as pd
import torch

import seaborn as sns

from torch import nn, optim
from torch.nn.functional import mse_loss

from tqdm import tqdm


class CLAMP(nn.Module):
    """Continual learning analysis via a model of performance"""
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
                                self.alg_memory_horizon, self.alg_experience_boost, self.task_difficulty)

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
                    self.sig[algo, task2, ind + 1] = (1 / (
                            1 + (torch.exp(-1 * self.result[algo, task2, ind + 1] / difficulty))) * 2) - 1

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

    def optimize(self, lx, target, adam_iters=100, adam_lr=0.01, lbfgs_iters=10):
        adam = optim.Adam(self.parameters(), lr=adam_lr)
        lbfgs = optim.LBFGS(self.parameters(), history_size=10, max_iter=4)

        self.target = target
        self.lx = lx
        self.losses = {'optimizer': [], 'step': [], 'loss': []}
        prediction = None

        # lbfgs evaluates prediction multiple times and needs a closure
        def closure():
            lbfgs.zero_grad()
            _prediction = self(lx)
            _loss = mse_loss(target, prediction)
            _loss.backward(retain_graph=True)
            return _loss

        for step in tqdm(range(adam_iters)):
            adam.zero_grad()

            prediction = self(lx)
            loss = mse_loss(target, prediction)
            loss.backward()
            adam.step()
            self.clamp()
            self.update('adam', step, loss.item())

        for step in tqdm(range(lbfgs_iters)):
            # loss = closure().item()
            lbfgs.step(closure)
            self.clamp()
            self.update('lbfgs', adam_iters+step, loss)

        self.prediction = prediction

    def clamp(self):
        with torch.no_grad():
            self.task_matrix.clamp_(-1.0, 1.0)
            self.task_difficulty.clamp_min_(0.0)
            self.alg_efficiency.clamp_min_(0.0)
            self.alg_memory_horizon.clamp_(0.0, 1.0)
            self.alg_experience_boost.clamp_min_(0.0)

    def update(self, optimizer, step, loss):
        self.losses['optimizer'].append(optimizer)
        self.losses['step'].append(step)
        self.losses['loss'].append(loss)

    def get_algo_results(self, algo_names=None):
        data = {'algorithm': range(self.nalgos),
                'efficiency': self.alg_efficiency.detach().numpy(),
                'retention': self.alg_memory_horizon.detach().numpy(),
                'expertise': self.alg_experience_boost.detach().numpy()}
        if algo_names is not None:
            data['algorithm'] = algo_names
        return pd.DataFrame(data)

    def get_task_transfer(self):
        return pd.DataFrame(self.task_matrix.detach().numpy())

    def plot_losses(self):
        sns.lineplot(x='step', y='loss', hue='optimizer', data=self.losses)


def test():
    torch.autograd.set_detect_anomaly(True)

    n_tasks = 5
    n_algos = 3

    np.random.seed(0)
    ntasks = n_tasks
    true_task_association = ((np.random.random((ntasks, ntasks)) * 2) - 1)
    true_task_difficulty = np.random.random(ntasks)
    for i in range(ntasks):
        true_task_association[i, i] = 1

    nalgos = n_algos
    true_algo_efficiency = np.random.random(nalgos)
    true_algo_memory = np.random.random(nalgos)
    true_algo_experience_bonus = np.random.random(nalgos)

    lx = []
    slices = list()
    order = np.random.randint(low=0, high=ntasks + 1, size=9)
    for i in order:
        before = len(lx)
        lx.extend([i])
        after = len(lx)
        slices.append([0, before, after, i])

    model = CLAMP(nalgos, ntasks)
    # model = IdealLearner(nalgos, ntasks)
    target = model.performance(ntasks, nalgos, true_task_association, lx, true_algo_efficiency, true_algo_memory,
                               true_algo_experience_bonus, true_task_difficulty)
    model.optimize(lx, target, adam_iters=10)


if __name__ == '__main__':
    test()
