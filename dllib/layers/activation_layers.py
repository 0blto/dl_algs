from ..utils import device
from .base import Layer

import math


class ReLULayer(Layer):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        self._mask = (x > 0).astype(xp.float32)
        return xp.maximum(0, x)

    def backward(self, dout):
        return dout * self._mask

    def parameters(self):
        return []

    def gradients(self):
        return []


class LeakyReLULayer(Layer):
    def __init__(self, negative_slope=0.2):
        self.negative_slope = negative_slope

    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        self._mask = (x > 0).astype(xp.float32)
        self._neg = (x <= 0).astype(xp.float32) * self.negative_slope
        return xp.where(x > 0, x, self.negative_slope * x)

    def backward(self, dout):
        return dout * (self._mask + self._neg)

    def parameters(self):
        return []

    def gradients(self):
        return []


class SigmoidLayer(Layer):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        self._out = 1.0 / (1.0 + xp.exp(-xp.clip(x, -500, 500)))
        return self._out

    def backward(self, dout):
        return dout * self._out * (1.0 - self._out)

    def parameters(self):
        return []

    def gradients(self):
        return []


class TanhLayer(Layer):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        self._out = xp.tanh(x)
        return self._out

    def backward(self, dout):
        xp = device.xp
        return dout * (1 - xp.square(self._out))

    def parameters(self):
        return []

    def gradients(self):
        return []


class GELULayer(Layer):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        self._x = x
        c = math.sqrt(2.0 / math.pi)
        self._tanh = xp.tanh(c * (x + 0.044715 * x ** 3))
        return 0.5 * x * (1 + self._tanh)

    def backward(self, dout):
        xp = device.xp
        x = self._x
        c = math.sqrt(2.0 / math.pi)
        t = self._tanh
        dx = 0.5 * (1 + t + x * (1 - t ** 2) * c * (1 + 3 * 0.044715 * x ** 2))
        return dout * dx

    def parameters(self):
        return []

    def gradients(self):
        return []


class SoftmaxLayer(Layer):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)
        exp_x = xp.exp(x - xp.max(x, axis=1, keepdims=True))
        self._out = exp_x / xp.sum(exp_x, axis=1, keepdims=True)
        return self._out

    def backward(self, dout):
        xp = device.xp
        return xp.ones_like(dout)

    def parameters(self):
        return []

    def gradients(self):
        return []


__all__ = [
    "ReLULayer", "LeakyReLULayer",
    "SigmoidLayer", "TanhLayer", "GELULayer", "SoftmaxLayer"
]
