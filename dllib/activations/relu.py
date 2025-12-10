from .base import Activation
from ..utils import device


class ReLU(Activation):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        return xp.maximum(0, x)
    
    def backward(self, output):
        xp = device.xp
        output = device.ensure_tensor(output)
        return (output > 0).astype(xp.float32)


__all__ = ["ReLU"]
