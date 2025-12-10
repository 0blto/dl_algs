from .base import Activation
from ..utils import device


class LeakyReLU(Activation):
    def __init__(self, negative_slope: float = 0.2):
        self.negative_slope = negative_slope
    
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        return xp.where(x > 0, x, self.negative_slope * x)
    
    def backward(self, output):
        xp = device.xp
        output = device.ensure_tensor(output)
        return xp.where(output > 0, 1.0, self.negative_slope)


__all__ = ["LeakyReLU"]

