import numpy as np
from torch.optim.optimizer import Optimizer


def annealing_cos(start, end, pct: float):
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out


class OneCycleCosine(object):
    def __init__(self,
                 optimizer: Optimizer,
                 total_steps: int,
                 warmup: float = 0.3,
                 lr_range=(1e-5, 1e-3),
                 moms_range=(0.85, 0.95)):

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.lr_range = lr_range
        self.moms_range = moms_range
        self.warmup = warmup
        self.final_div = 24 * 1e4

        self._lr = None
        self._momentum = None

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

        if self.cur_step < self.warmup*self.total_steps:
            steps = self.cur_step
            phase_steps = self.warmup*self.total_steps

            lr = annealing_cos(self.lr_range[0], self.lr_range[1], steps/phase_steps)
            momentum = annealing_cos(self.moms_range[1], self.moms_range[0], steps/phase_steps)
        else:
            steps = self.warmup*self.total_steps - self.cur_step
            phase_steps = self.total_steps - self.warmup*self.total_steps

            lr = annealing_cos(self.lr_range[1], self.lr_range[1]/self.final_div, steps/phase_steps)
            momentum = annealing_cos(self.moms_range[0], self.moms_range[1], steps/phase_steps)

        self.lr = lr
        self.momentum = momentum


class OneCycleCosineAdam(OneCycleCosine):
    @property
    def momentum(self):
        return self.optimizer.param_groups[0]['betas'][0]

    @momentum.setter
    def momentum(self, m):
        self.optimizer.param_groups[0]['betas'] = (m, self.optimizer.param_groups[0]['betas'][1])
