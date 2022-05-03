import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from tqdm import tqdm


class CLAMP(nn.Module):
    """Continual learning analysis via a model of performance"""
    def __init__(self, nalgos: int, ntasks: int, seed: int = 0):
        """

        Parameters
        ----------
        nalgos: Number of learning algorithms to represent
        ntasks: Number of different tasks the algorithms will possibly be exposed to
        seed: Seed for the random number generator (0 by default)
        """
        super().__init__()
        # general parameters
        self.ntasks = ntasks
        self.nalgos = nalgos
        self.task_difficulty_scale = 1

        # initialize parameters that can be estimated
        torch.manual_seed(seed)
        self.task_matrix = nn.Parameter(
            torch.tensor(np.random.random((self.ntasks, self.ntasks)) * 2 - 1, requires_grad=True))
        self.task_difficulty = nn.Parameter(torch.tensor(np.random.random((self.ntasks,)), requires_grad=True))
        self.alg_efficiency = nn.Parameter(torch.tensor(np.random.normal(0.5, 0.02, size=nalgos), requires_grad=True))
        self.alg_memory_horizon = nn.Parameter(torch.tensor(np.random.normal(0.5, 0.1, size=nalgos),
                                                            requires_grad=True))
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

    def performance(self, ntasks: int, nalgos: int, task_matrix: torch.Tensor, lx: list, algo_efficiency: torch.Tensor,
                    algo_memory: torch.Tensor, algo_experience_boost: torch.Tensor, task_difficulty: torch.Tensor):
        """
        Given the current state of the learned algorithm and task parameters, compute the performance of the algorithms
            on a given curriculum (set of learning experiences, or LXs). !!NOTE!! Performance curves can be unstable
            when applying the exponential near zero, which can cause the optimization to fail.
        Parameters
        ----------
        nalgos: Number of learning algorithms to represent
        ntasks: Number of different tasks the algorithms will possibly be exposed to
        task_matrix: Task similarity matrix. A 2D tensor of shape ntasks x ntasks that represents how similar task i is
            to task j; diagonal entries should be 1 (i.e. a task is perfectly similar to itself).
        lx: List of integer values representing a curriculum. Each value in the list represents a "Learning Experience"
            or LX, which is the fundamental unit of data an agent receives on which it can learn.
        algo_efficiency: Efficiency parameter of each learning algorithm. A 1D tensor of size nalgos.
        algo_memory: Memory horizon parameter of each learning algorithm. A 1D tensor of size nalgos.
        algo_experience_boost: Experience boost parameter of each learning algorithm. A 1D tensor of size nalgos.
        task_difficulty: Difficulty value for each task. A 1D tensor of size ntasks.

        Returns - Performance data as a tensor
        -------

        """
        self.result = torch.tensor(np.zeros((nalgos, ntasks, len(lx) + 1)))
        self.sig = torch.tensor(np.zeros((nalgos, ntasks, len(lx) + 1)))

        for algo in range(nalgos):
            for ind, task in enumerate(lx):
                for task2 in range(task_matrix.shape[0]):
                    prior_exp = torch.sum(self.result[algo, task2, ind:ind + 1]) * algo_memory[algo]
                    prior_perf = torch.sum(self.sig[algo, task, ind:ind + 1])
                    self.result[algo, task2, ind + 1] = (prior_exp
                                                         + task_matrix[task, task2] * algo_efficiency[algo]
                                                         + (task_matrix[task, task2] * prior_perf
                                                            * algo_experience_boost[algo]))

                    difficulty = task_difficulty[task2] * self.task_difficulty_scale
                    self.sig[algo, task2, ind + 1] = (1 / (
                            1 + (torch.exp(-1 * self.result[algo, task2, ind + 1] / difficulty))) * 2) - 1

        return self.sig

    def optimize(self, lx: list, target: torch.Tensor, adam_iters: int = 1000, adam_lr: float = 0.01,
                 lbfgs_iters: int = 0):
        """

        Parameters
        ----------
        lx: List of integer values representing a curriculum. Each value in the list represents a "Learning Experience"
            or LX, which is the fundamental unit of data an agent receives on which it can learn.
        target: Actual performance data from each algorithm.
        adam_iters: Number of iterations to run the Adam optimizer. Default is 1000 iterations.
        adam_lr: Adam optimizer learning rate. Default is 0.01.
        lbfgs_iters: Number of iterations to run the LBFGS optimizer after the Adam optimizer iterations. Default is 0.
            This is an experimental addition to this module, and is not well tested.

        -------

        """
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
            _loss = mse_loss(target, _prediction)
            _loss.backward()
            self.clamp()
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
            loss = lbfgs.step(closure)
            self.clamp()
            self.update('lbfgs', adam_iters+step, loss)

        self.prediction = prediction

    def clamp(self):
        """
        Clamp the values of the parameters to stay within reasonable bounds.
        -------

        """
        with torch.no_grad():
            self.task_matrix.clamp_(-1.0, 1.0)
            self.task_difficulty.clamp_min_(0.0)
            self.alg_efficiency.clamp_min_(0.0)
            self.alg_memory_horizon.clamp_(0.0, 1.0)
            self.alg_experience_boost.clamp_min_(0.0)
            self.result.clamp_min_(0.0)

    def update(self, optimizer, step, loss):
        """
        Record current state of optimization
        Parameters
        ----------
        optimizer: Optimizer in its current state
        step: Current step
        loss: Current loss

        -------
        """
        self.losses['optimizer'].append(optimizer)
        self.losses['step'].append(step)
        self.losses['loss'].append(loss)

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
