from typing import Sequence


from .base import NeuralNetwork
from ..activations import LeakyReLU
from ..layers import ConvLayer, DenseLayer, FlattenLayer, MaxPoolLayer, DropoutLayer
from ..losses import CrossEntropyLoss
from ..optimizers import Adam
from ..trainers import SimpleTrainer


class CNN(NeuralNetwork):
    def __init__(self, input_depth: int = 1, num_classes: int = 10):
        leaky_relu = LeakyReLU(negative_slope=0.1)
        architecture: Sequence = [
            ConvLayer(6, 5, input_depth, padding=2, activation=leaky_relu),
            MaxPoolLayer(2, 2),
            ConvLayer(16, 5, 6, activation=leaky_relu),
            MaxPoolLayer(2, 2),
            FlattenLayer(),
            DenseLayer(16 * 6 * 6, 100, activation=leaky_relu),
            DropoutLayer(0.3),
            DenseLayer(100, 64, activation=leaky_relu),
            DropoutLayer(0.5),
            DenseLayer(64, num_classes),
        ]
        super().__init__(
            architecture,
            optimizer=Adam(0.0005),
            trainer=SimpleTrainer(CrossEntropyLoss(num_classes))
        )


__all__ = ["CNN"]

