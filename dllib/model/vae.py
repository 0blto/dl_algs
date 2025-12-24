from typing import Sequence

from .base import NeuralNetwork
from ..layers import (
    ConvLayer, FlattenLayer, DenseLayer, BatchNormLayer, ReshapeLayer
)
from ..layers.activation_layers import LeakyReLULayer, ReLULayer, TanhLayer, SigmoidLayer
from ..layers.conv_transpose_layer import ConvTransposeLayer
from ..optimizers import Adam
from ..utils import device


class VAEEncoder(NeuralNetwork):
    def __init__(self, input_depth=1, latent_dim=32):
        self.latent_dim = latent_dim

        self.layers = [
            ConvLayer(input_depth, 32, 4, stride=2, padding=1),
            BatchNormLayer(32),
            ReLULayer(),

            ConvLayer(32, 64, 4, stride=2, padding=1),
            BatchNormLayer(64),
            ReLULayer(),

            ConvLayer(64, 128, 4, stride=2, padding=1),
            BatchNormLayer(128),
            ReLULayer(),

            FlattenLayer(),
            DenseLayer(128 * 3 * 3, 512),
            ReLULayer(),

            DenseLayer(512, 2 * latent_dim)
        ]

        super().__init__(self.layers, optimizer=Adam(1e-4, beta1=0.5))

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)

        mu = h[:, :self.latent_dim]
        logvar = device.xp.clip(h[:, self.latent_dim:], -5, 5)

        eps = device.xp.random.randn(*mu.shape).astype(mu.dtype)
        std = device.xp.exp(0.5 * logvar)
        z = mu + std * eps

        self._cache = (mu, logvar, eps, std)
        return mu, logvar, z

    def backward(self, dz):
        mu, logvar, eps, std = self._cache
        xp = device.xp

        dmu = dz
        dlogvar = dz * eps * 0.5 * std

        dh = xp.concatenate([dmu, dlogvar], axis=1)

        for layer in reversed(self.layers):
            dh = layer.backward(dh)

        return dh



class VAEDecoder(NeuralNetwork):
    def __init__(self, output_depth=1, latent_dim=32):
        layers = [
            DenseLayer(latent_dim, 512),
            ReLULayer(),

            DenseLayer(512, 128 * 3 * 3),
            ReLULayer(),

            ReshapeLayer((128, 3, 3)),

            ConvTransposeLayer(128, 64, 4, stride=2, padding=1),  # 3 → 6
            BatchNormLayer(64),
            ReLULayer(),

            ConvTransposeLayer(64, 32, 4, stride=2, padding=1),   # 6 → 12
            BatchNormLayer(32),
            ReLULayer(),

            ConvTransposeLayer(32, output_depth, 4, stride=2, padding=1),  # 12 → 24
            TanhLayer()
        ]

        super().__init__(layers, optimizer=Adam(1e-4, beta1=0.5))


_all__ = [
    "VAEEncoder", "VAEDecoder"
]
