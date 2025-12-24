from .base import Layer
from ..utils import device


class LayerNorm(Layer):
    def __init__(self, feature_dim, eps=1e-5, dtype=None):
        xp = device.xp
        dtype = dtype or xp.float32

        self.gamma = xp.ones(feature_dim, dtype=dtype)
        self.beta = xp.zeros(feature_dim, dtype=dtype)
        self.eps = eps

        self.x = None
        self.mean = None
        self.var = None
        self.x_hat = None

        self.d_gamma = None
        self.d_beta = None

    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)

        self.x = x
        self.mean = xp.mean(x, axis=-1, keepdims=True)
        self.var = xp.var(x, axis=-1, keepdims=True)

        self.x_hat = (x - self.mean) / xp.sqrt(self.var + self.eps)

        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)

        N = self.x.shape[-1]

        self.d_gamma = xp.sum(dout * self.x_hat, axis=tuple(range(dout.ndim - 1)))
        self.d_beta = xp.sum(dout, axis=tuple(range(dout.ndim - 1)))

        dx_hat = dout * self.gamma
        std_inv = 1.0 / xp.sqrt(self.var + self.eps)

        dx = (1.0 / N) * std_inv * (
            N * dx_hat
            - xp.sum(dx_hat, axis=-1, keepdims=True)
            - self.x_hat * xp.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)
        )

        return dx

    def parameters(self):
        return [self.gamma, self.beta]

    def gradients(self):
        return [self.d_gamma, self.d_beta]


__all__ = ["LayerNorm"]