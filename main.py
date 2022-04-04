from torch.optim.lr_scheduler import MultiStepLR

from simulateur import Simulateur
import torch as t
import torch.optim as op
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

target = t.tensor([17.5, 0])
x0 = t.tensor([0.0, 0.0], requires_grad=True)
v0 = t.tensor([10.0, 10.0], requires_grad=True)
optimiseur = op.Adam([x0, v0], lr=.01)
lrAdjustment = MultiStepLR(optimiseur, milestones=[50], gamma=0.01)
losses = []

for _ in tqdm(range(200)):
    sim = Simulateur(x0, v0)
    sim.run()
    loss = ((x0 - target)**2).sum()
    optimiseur.zero_grad()
    loss.backward()
    optimiseur.step()
    lrAdjustment.step()
    losses.append(loss.item())

#plt.plot(losses)
#plt.show()

print(sim.x0)

