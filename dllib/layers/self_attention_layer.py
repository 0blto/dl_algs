import math
from .base import Layer
from ..utils import device

class SelfAttentionLayer(Layer):
    def __init__(self, d_model):
        xp = device.xp
        self.d_model = d_model
        self.sqrt_dk = math.sqrt(d_model)
        self.W_Q = xp.random.randn(d_model, d_model) * 0.01
        self.W_K = xp.random.randn(d_model, d_model) * 0.01
        self.W_V = xp.random.randn(d_model, d_model) * 0.01
        self.W_O = xp.random.randn(d_model, d_model) * 0.01
        self.dW_Q = None
        self.dW_K = None
        self.dW_V = None
        self.dW_O = None
        self.X = None
        self.Q = None
        self.K = None
        self.V = None
        self.scores = None
        self.A = None

    def forward(self, X):
        xp = device.xp
        X = device.ensure_tensor(X)
        self.X = X

        self.Q = X @ self.W_Q
        self.K = X @ self.W_K
        self.V = X @ self.W_V

        self.scores = self.Q @ self.K.transpose(0, 2, 1) / self.sqrt_dk
        exp_scores = xp.exp(self.scores - xp.max(self.scores, axis=-1, keepdims=True))
        self.A = exp_scores / xp.sum(exp_scores, axis=-1, keepdims=True)

        out = self.A @ self.V
        out = out @ self.W_O
        return out

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)

        out_pre = self.A @ self.V
        self.dW_O = xp.tensordot(out_pre, dout, axes=([0, 1], [0, 1]))
        d_out_pre = dout @ self.W_O.T

        dA = d_out_pre @ self.V.transpose(0, 2, 1)
        dV = self.A.transpose(0, 2, 1) @ d_out_pre

        d_scores = self.A * (dA - xp.sum(dA * self.A, axis=-1, keepdims=True))

        dQ = d_scores @ self.K / self.sqrt_dk
        dK = d_scores.transpose(0, 2, 1) @ self.Q / self.sqrt_dk

        self.dW_Q = xp.tensordot(self.X, dQ, axes=([0, 1], [0, 1]))
        self.dW_K = xp.tensordot(self.X, dK, axes=([0, 1], [0, 1]))
        self.dW_V = xp.tensordot(self.X, dV, axes=([0, 1], [0, 1]))

        dX = dQ @ self.W_Q.T + dK @ self.W_K.T + dV @ self.W_V.T
        return dX

    def parameters(self):
        return [self.W_Q, self.W_K, self.W_V, self.W_O]

    def gradients(self):
        return [self.dW_Q, self.dW_K, self.dW_V, self.dW_O]

__all__ = ["SelfAttentionLayer"]
