from .base import Layer
from ..utils import device


class EmbeddingLayer(Layer):
    def __init__(self, vocab_size: int, embed_dim: int, dtype=None):
        xp = device.xp
        dtype = dtype or xp.float32

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.weights = xp.random.randn(vocab_size, embed_dim).astype(dtype) * 0.01

        self.last_indices = None

        self.d_weights = None

    def forward(self, input_data):
        xp = device.xp
        input_data = device.ensure_tensor(input_data)

        if input_data.dtype.kind not in ("i", "u"):
            raise ValueError("EmbeddingLayer expects integer indices")

        self.last_indices = input_data

        return self.weights[input_data]

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)

        self.d_weights = xp.zeros_like(self.weights)

        xp.add.at(self.d_weights, self.last_indices, dout)

        return None

    def parameters(self):
        return [self.weights]

    def gradients(self):
        return [self.d_weights]


__all__ = ["EmbeddingLayer"]