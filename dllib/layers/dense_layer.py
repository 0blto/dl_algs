from .base import Layer
from ..utils import device


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, dtype=None):
        xp = device.xp
        limit = (6.0 / (input_size + output_size)) ** 0.5
        dtype = dtype or xp.float32
        self.weights = xp.random.uniform(-limit, limit, (input_size, output_size)).astype(dtype)
        self.biases = xp.zeros(output_size, dtype=dtype)
        self.last_input = None
        self.last_z = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, input_data):
        input_data = device.ensure_tensor(input_data)
        self.last_input = input_data
        self.last_z = input_data @ self.weights + self.biases
        return self.last_z  # возвращаем чистую линейную комбинацию

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)
        self.d_weights = self.last_input.T @ dout
        self.d_biases = xp.sum(dout, axis=0)
        return dout @ self.weights.T

    def parameters(self):
        return [self.weights, self.biases]

    def gradients(self):
        return [self.d_weights, self.d_biases]


__all__ = ["DenseLayer"]
