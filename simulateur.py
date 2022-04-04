import torch as t
import torch.optim as op
import matplotlib.pyplot as plt
import numpy as np

# création de la classe Simulateur
class Simulateur:
    def __init__(self, x0 = t.tensor([0, 0]), v0 = t.tensor([10, 10]), g = 9.81):
        self.x0 = x0
        self.v0 = v0
        self.history = [self.x0.tolist()]
        self.acceleration = t.tensor([0, -g])

    def run(self, max_step = 50000):
        for _ in range(max_step):
            xprev = self.x0.clone()
            self.step_update()
            if (xprev[1] > 0) and (self.x0[1] <= 0):
                return

    def step_update(self, delta_t = 0.001):
        self.x0 = self.x0 + self.v0 * delta_t + 1/2 * self.acceleration * delta_t ** 2
        self.v0 = self.v0 + self.acceleration * delta_t
        self.history.append(self.x0.tolist())

    def show(self):
        plt.scatter(np.array(self.history)[:, 0], np.array(self.history)[:, 1])
        plt.show()

