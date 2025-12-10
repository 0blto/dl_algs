class Optimizer:
    def step(self, params, grads):
        raise NotImplementedError

__all__ = ["Optimizer"]