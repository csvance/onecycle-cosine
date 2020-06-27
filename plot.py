import matplotlib.pyplot as plt
from torch.optim import AdamW
import torch.nn as nn
from onecycle import OneCycleCosineAdam


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


model = ToyModel()
optimizer = AdamW(model.parameters(), lr=0.01)

N = 1000
sched = OneCycleCosineAdam(optimizer, total_steps=1000)

m = []
l = []

for n in range(0, N):
    m.append(sched.momentum)
    l.append(sched.lr)
    sched.step()

plt.plot(l)
plt.title('Learning Rate')
plt.grid()
plt.xlabel('step')
plt.show()

plt.plot(m)
plt.title('Momentum')
plt.grid()
plt.xlabel('step')
plt.show()
