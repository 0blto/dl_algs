from .base import Layer
from ..utils import device


class ReshapeLayer(Layer):
    def __init__(self, shape):
        self.shape = shape
        self.last_input_shape = None

    def forward(self, x):
        x = device.ensure_tensor(x)
        self.last_input_shape = x.shape

        batch = x.shape[0]
        return x.reshape((batch, *self.shape))

    def backward(self, dout):
        dout = device.ensure_tensor(dout)
        return dout.reshape(self.last_input_shape)

    def parameters(self):
        return []

    def gradients(self):
        return []


__all__ = ["ReshapeLayer"]

