from .base import Layer
from ..utils import device


class ResizeLayer(Layer):
    def __init__(self, scale=2):
        self.scale = scale
        self.last_input_shape = None

    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)

        self.last_input_shape = x.shape
        b, c, h, w = x.shape
        s = self.scale

        x = x.reshape(b, c, h, 1, w, 1)

        x = xp.repeat(x, s, axis=3)
        x = xp.repeat(x, s, axis=5)

        return x.reshape(b, c, h * s, w * s)

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)

        b, c, h, w = self.last_input_shape
        s = self.scale

        dout = dout.reshape(b, c, h, s, w, s)

        dx = xp.sum(dout, axis=(3, 5))

        return dx

    def parameters(self):
        return []

    def gradients(self):
        return []


__all__ = ["ResizeLayer"]
