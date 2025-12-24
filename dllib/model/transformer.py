from typing import List

import numpy as np

from dllib.layers import TransformerLayer, DenseLayer, Layer, DropoutLayer, EmbeddingLayer
from dllib.layers import MeanPoolingLayer
from dllib.losses import CrossEntropyLoss
from dllib.model import NeuralNetwork
from dllib.optimizers import Adam
from dllib.trainers import SimpleTrainer

class Transformer(NeuralNetwork):
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads=4, ff_hidden_dim=64,
                 num_classes=2, num_layers=2, dropout=0.1):

        layers: List[Layer] = [
            EmbeddingLayer(vocab_size, d_model)
        ]
        for _ in range(num_layers):
            layers.append(TransformerLayer(d_model, num_heads, ff_hidden_dim))
            layers.append(DropoutLayer(dropout))
        layers.append(MeanPoolingLayer())
        layers.append(DropoutLayer(dropout))
        layers.append(DenseLayer(d_model, num_classes))

        super().__init__(
            layers,
            optimizer=Adam(0.00001),
            trainer=SimpleTrainer(self, CrossEntropyLoss(num_classes))
        )


__all__ = ["Transformer"]