from functools import reduce
from typing import List, Optional, Sequence

from ..layers import Layer
from ..optimizers import Optimizer, SGD
from ..trainers import Trainer
from ..utils import device, TrainingModeMixin
from ..utils.decorators import around, before


class NeuralNetwork:
    def __init__(self, architecture: Sequence[Layer], optimizer: Optional[Optimizer] = None, trainer: Optional[Trainer] = None):
        self.layers: List[Layer] = list(architecture)
        self.optimizer: Optimizer = optimizer if optimizer else SGD(lr=0.01)
        self.trainer: Optional[Trainer] = trainer
        self._training = False

    @property
    def training(self) -> bool: return self._training

    @training.setter
    def training(self, mode: bool):
        self._training = mode
        for layer in self.layers: layer.set_training_mode(mode) if isinstance(layer, TrainingModeMixin) else ""

    def forward(self, x): return reduce(lambda out, layer: layer.forward(out), self.layers, x)

    def backward(self, dout): return reduce(lambda grad, layer: layer.backward(grad), reversed(self.layers), dout)

    def update_weights(self):
        self.optimizer.step(
            [p for layer in self.layers for p in layer.parameters()],
            [g for layer in self.layers for g in layer.gradients()]
        )

    @around(
        before_fn=lambda self: setattr(self, "training", True),
        after_fn=lambda self: setattr(self, "training", False),
    )
    def train(self, *args, **kwargs):
        if self.trainer is None: raise ValueError("Trainer не передан в модель")
        history = self.trainer.train(*args, **kwargs)
        return history

    @around(
        before_fn=lambda self: setattr(self, "_prev_training", self.training) or setattr(self, "training", False),
        after_fn=lambda self: setattr(self, "training", getattr(self, "_prev_training"))
    )
    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x):
        xp = device.xp
        return xp.argmax(self.predict_proba(x), axis=1)


__all__ = ["NeuralNetwork"]
