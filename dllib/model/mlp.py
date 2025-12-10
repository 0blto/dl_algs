from typing import Sequence

from sympy.vector import Cross

from .base import NeuralNetwork
from ..activations import Sigmoid
from ..layers import DenseLayer, Layer
from ..losses import CrossEntropyLoss
from ..optimizers import SGDWithMomentum
from ..trainers import SimpleTrainer


class MLP(NeuralNetwork):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ) -> None:
        sigmoid = Sigmoid()
        architecture: Sequence[Layer] = [
            DenseLayer(in_features, 256, activation=sigmoid),
            DenseLayer(256, 128, activation=sigmoid),
            DenseLayer(128, num_classes),
        ]
        super().__init__(
            architecture=architecture,
            optimizer=SGDWithMomentum(lr=0.01, momentum=0.9),
            trainer=SimpleTrainer(CrossEntropyLoss(num_classes))
        )


__all__ = ["MLP"]
