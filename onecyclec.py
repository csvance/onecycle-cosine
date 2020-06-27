import numpy as np
from torch.optim.optimizer import Optimizer


# Borrowed from FastAI
def annealing_cos(start, end, pct: float):
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycleCosine(object):
    def __init__(self,
                 optimizer: Optimizer,
                 num_steps: int,
                 warmup: float = 0.3,
                 plateau: float = 0.,
                 winddown: float = 0.7,
                 lr_range=(1e-5, 1e-3),
                 momentum_range=(0.85, 0.95)):
        """
        OneCycleCosine version for optimizers which have a momentum parameter

        :param optimizer: PyTorch Optimizer object
        :param num_steps: Total number of training steps, ie epochs*len(dataset_train)//batch_size
        :param warmup: Proportion of time to spend in the cosine annealed warmup. Will be normalized.
        :param plateau: Proportion of time to spend on the maximum learning rate plateau. Will be normalized.
        :param winddown: Proportion of time to spend in the cosine annealed windown. Will be normalized.
        :param lr_range: (min_lr, max_lr)
        :param momentum_range: (min_momentum, max_momentum)
        """

        self.optimizer = optimizer

        self.num_steps = num_steps
        self.lr_range = lr_range
        self.momentum_range = momentum_range

        # Phases
        self.warmup = warmup / (warmup + plateau + winddown)
        self.plateau = plateau / (warmup + plateau + winddown)
        self.winddown = winddown / (warmup + plateau + winddown)

        self.cur_step = -1
        self.step()

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @lr.setter
    def lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    @property
    def momentum(self):
        return self.optimizer.param_groups[0]['momentum']

    @momentum.setter
    def momentum(self, m):
        self.optimizer.param_groups[0]['momentum'] = m

    def step(self):
        self.cur_step += 1

        if 0 <= self.cur_step < self.warmup * self.num_steps:
            steps = self.cur_step
            phase_steps = self.warmup * self.num_steps

            lr = annealing_cos(self.lr_range[0], self.lr_range[1], steps / phase_steps)
            momentum = annealing_cos(self.momentum_range[1], self.momentum_range[0], steps / phase_steps)
        elif self.warmup * self.num_steps <= self.cur_step < (self.warmup + self.plateau) * self.num_steps:
            lr = self.lr_range[1]
            momentum = self.momentum_range[0]
        elif (self.warmup + self.plateau) * self.num_steps <= self.cur_step < self.num_steps:

            steps = self.cur_step - (self.warmup + self.plateau) * self.num_steps
            phase_steps = self.winddown * self.num_steps

            lr = annealing_cos(self.lr_range[1], self.lr_range[1] / 24e4, steps / phase_steps)
            momentum = annealing_cos(self.momentum_range[0], self.momentum_range[1], steps / phase_steps)
        else:
            lr = 0.
            momentum = self.momentum_range[1]

        self.lr = lr
        self.momentum = momentum


class OneCycleCosineAdam(OneCycleCosine):
    """
    OneCycleCosine version for Adam based optimizers which have a betas parameter tuple
    """
    @property
    def momentum(self):
        return self.optimizer.param_groups[0]['betas'][0]

    @momentum.setter
    def momentum(self, m):
        self.optimizer.param_groups[0]['betas'] = (m, self.optimizer.param_groups[0]['betas'][1])
