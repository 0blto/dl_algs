from .base import Activation
from ..utils import device


class Sigmoid(Activation):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        return 1.0 / (1.0 + xp.exp(-xp.clip(x, -500, 500)))

    def backward(self, output):
        output = device.ensure_tensor(output)
        s = self.forward(output)
        return s * (1.0 - s)


__all__ = ["Sigmoid"]
