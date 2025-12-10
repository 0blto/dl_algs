from .base import Activation
from ..utils import device


class Tanh(Activation):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        return xp.tanh(x)

    def backward(self, output):
        xp = device.xp
        output = device.ensure_tensor(output)
        return 1 - xp.square(xp.tanh(output))


__all__ = ["Tanh"]
