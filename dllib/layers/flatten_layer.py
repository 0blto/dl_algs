from .base import Layer
from ..utils import device


class FlattenLayer(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, input_data):
        input_data = device.ensure_tensor(input_data)
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, dout):
        return device.ensure_tensor(dout).reshape(self.input_shape)

    def parameters(self):
        return []

    def gradients(self):
        return []


__all__ = ["FlattenLayer"]

