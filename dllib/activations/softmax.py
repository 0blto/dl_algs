from .base import Activation
from ..utils import device


class Softmax(Activation):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        exp_x = xp.exp(x - xp.max(x, axis=1, keepdims=True))
        return exp_x / xp.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, output):
        xp = device.xp
        output = device.ensure_tensor(output)
        return xp.ones_like(output)


__all__ = ["Softmax"]

