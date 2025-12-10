from .base import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads): p -= self.lr * g


__all__ = ['SGD']
