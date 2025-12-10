from .base import Layer
from ..utils import device


class LastTimestepLayer(Layer):
    def forward(self, input_data):
        input_data = device.ensure_tensor(input_data)
        self.last_input_shape = input_data.shape
        return input_data[:, -1, :]
    
    def backward(self, dout):
        xp = device.xp
        dout = xp.asarray(dout, dtype=xp.float32)
        
        d_input = xp.zeros(self.last_input_shape, dtype=xp.float32)
        d_input[:, -1, :] = dout
        return d_input

    def gradients(self):
        return []

    def parameters(self):
        return []


__all__ = ["LastTimestepLayer"]

