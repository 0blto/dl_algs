from functools import reduce
from typing import List, Optional, Sequence

from ..layers import Layer
from ..optimizers import Optimizer, SGD
from ..trainers import Trainer
from ..utils import device, TrainingModeMixin


class NeuralNetwork:
    def __init__(self, architecture: Sequence[Layer], optimizer: Optional[Optimizer] = None, trainer: Optional[Trainer] = None):
        self.layers: List[Layer] = list(architecture)
        self.optimizer: Optimizer = optimizer if optimizer else SGD(lr=0.01)
        self.trainer: Optional[Trainer] = trainer
        self._training = True

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, mode: bool):
        self._training = mode
        for layer in self.layers: layer.set_training_mode(mode) if isinstance(layer, TrainingModeMixin) else ""

    def forward(self, x):
        return reduce(lambda out, layer: layer.forward(out), self.layers, x)

    def backward(self, dout):
        return reduce(lambda grad, layer: layer.backward(grad), reversed(self.layers), dout)

    def update_weights(self):
        self.optimizer.step(
            [p for layer in self.layers for p in layer.parameters()],
            [g for layer in self.layers for g in layer.gradients()]
        )

    def predict(self, x):
        xp = device.xp
        _training = False
        return xp.argmax(self.forward(x), axis=1)

    def train(self, *args, **kwargs):
        if self.trainer is None: raise ValueError("Trainer не передан в модель")
        _training = True
        history = self.trainer.train(self, *args, **kwargs)
        return history


__all__ = ["NeuralNetwork"]
