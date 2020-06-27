import matplotlib.pyplot as plt
from torch.optim import AdamW
import torch.nn as nn
from onecyclec import OneCycleCosineAdam


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


model = ToyModel()
optimizer = AdamW(model.parameters(), lr=0.01)

WARMUP = 0.3
PLATEAU = 0.3
WINDDOWN = 0.7

N = 1000
sched = OneCycleCosineAdam(optimizer,
                           warmup=WARMUP,
                           plateau=PLATEAU,
                           winddown=WINDDOWN,
                           num_steps=N)

momentum = []
lr = []

for n in range(0, N):
    momentum.append(sched.momentum)
    lr.append(sched.lr)
    sched.step()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_xlabel('step')
ax1.set_ylabel('learning rate', color='tab:blue')
ax1.plot(lr, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2.set_ylabel('momentum', color='green')
ax2.plot(momentum, color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('warmup = %.1f plateau = %.1f winddown = %.1f' % (WARMUP, PLATEAU, WINDDOWN))
plt.savefig('sched.png', dpi=100, tight=True)
plt.show()
