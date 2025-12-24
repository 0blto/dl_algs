from dllib.model.vae import VAEDecoder
from typing import Sequence
from .base import NeuralNetwork
from ..layers import ConvLayer, DenseLayer, FlattenLayer
from ..layers.activation_layers import LeakyReLULayer, SigmoidLayer
from ..optimizers import Adam

class Generator(VAEDecoder): pass


class Discriminator(NeuralNetwork):
    def __init__(self, input_depth: int = 1, out_features: int = 1):
        architecture: Sequence = [
            ConvLayer(input_depth, 32, 4, stride=2, padding=1),
            LeakyReLULayer(0.2),
            ConvLayer(32, 64, 4, stride=2, padding=1),
            LeakyReLULayer(0.2),
            ConvLayer(64, 128, 4, stride=2, padding=1),
            LeakyReLULayer(0.2),
            FlattenLayer(),
            DenseLayer(1152, out_features),
            SigmoidLayer(),
        ]

        super().__init__(architecture, optimizer=Adam(0.00005, beta1=0.5))


__all__ = [
    "Generator", "Discriminator",
]

