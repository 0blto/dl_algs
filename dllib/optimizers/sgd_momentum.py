from .base import Optimizer
from ..utils import device


class SGDWithMomentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None: self.velocities = [device.xp.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] + g
            p -= self.lr * self.velocities[i]


__all__ = ['SGDWithMomentum']
