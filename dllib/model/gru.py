from typing import Sequence

from .base import NeuralNetwork
from ..layers import DenseLayer, GRULayer, LastTimestepLayer
from ..losses import MSELoss
from ..optimizers import Adam
from ..trainers import SimpleTrainer


class GRU(NeuralNetwork):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        architecture: Sequence = [
            GRULayer(input_size, hidden_size),
            LastTimestepLayer(),
            DenseLayer(hidden_size, output_size),
        ]
        super().__init__(
            architecture,
            Adam(0.00001),
            SimpleTrainer(self, MSELoss())
        )


__all__ = ["GRU"]

