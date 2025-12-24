from typing import Sequence


from .base import NeuralNetwork
from ..activations import LeakyReLU
from ..layers import ConvLayer, DenseLayer, FlattenLayer, MaxPoolLayer, DropoutLayer
from ..layers.activation_layers import LeakyReLULayer
from ..losses import CrossEntropyLoss
from ..optimizers import Adam
from ..trainers import SimpleTrainer


class CNN(NeuralNetwork):
    def __init__(self, input_depth: int = 1, num_classes: int = 10):
        architecture: Sequence = [
            ConvLayer(input_depth, 6, 5, padding=2),
            LeakyReLULayer(negative_slope=0.1),
            MaxPoolLayer(2, 2),
            ConvLayer(6, 16, 5),
            LeakyReLULayer(negative_slope=0.1),
            MaxPoolLayer(2, 2),
            FlattenLayer(),
            DenseLayer(16 * 6 * 6, 100),
            LeakyReLULayer(negative_slope=0.1),
            DropoutLayer(0.3),
            DenseLayer(100, 64),
            LeakyReLULayer(negative_slope=0.1),
            DropoutLayer(0.5),
            DenseLayer(64, num_classes),
        ]
        super().__init__(
            architecture,
            optimizer=Adam(0.0005),
            trainer=SimpleTrainer(self, CrossEntropyLoss(num_classes))
        )


__all__ = ["CNN"]

