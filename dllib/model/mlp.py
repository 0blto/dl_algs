from typing import Sequence
from .base import NeuralNetwork
from ..layers import DenseLayer, Layer
from ..layers.activation_layers import SigmoidLayer
from ..losses import CrossEntropyLoss
from ..optimizers import SGDWithMomentum
from ..trainers import SimpleTrainer


class MLP(NeuralNetwork):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ) -> None:
        architecture: Sequence[Layer] = [
            DenseLayer(in_features, 256),
            SigmoidLayer(),
            DenseLayer(256, 128),
            SigmoidLayer(),
            DenseLayer(128, num_classes),
        ]
        super().__init__(
            architecture=architecture,
            optimizer=SGDWithMomentum(lr=0.01, momentum=0.9),
            trainer=SimpleTrainer(self, CrossEntropyLoss(num_classes))
        )


__all__ = ["MLP"]
