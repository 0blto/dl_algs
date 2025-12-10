from .base import Layer
from ..utils import device
from ..utils.training import TrainingModeMixin


class DropoutLayer(Layer, TrainingModeMixin):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, input_data):
        xp = device.xp
        input_data = device.ensure_tensor(input_data)
        if self.training:
            self.mask = xp.random.rand(*input_data.shape) > self.p
            output = input_data * self.mask / (1.0 - self.p)
            return output
        else: return input_data

    def backward(self, dout):
        return dout * self.mask / (1.0 - self.p) if self.training else dout

    def parameters(self):
        return []

    def gradients(self):
        return []

__all__ = ["DropoutLayer"]