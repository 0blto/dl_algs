from typing import Sequence

from .base import NeuralNetwork
from ..layers import DenseLayer, LastTimestepLayer, RNNLayer
from ..losses import MSELoss
from ..optimizers import Adam
from ..trainers import SimpleTrainer


class RNN(NeuralNetwork):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        architecture: Sequence = [
            RNNLayer(input_size, hidden_size),
            LastTimestepLayer(),
            DenseLayer(hidden_size, output_size)
        ]
        super().__init__(
            architecture,
            Adam(0.0001),
            SimpleTrainer(MSELoss())
        )


__all__ = ["RNN"]

