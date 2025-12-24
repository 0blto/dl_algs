import math
from .base import Layer
from ..utils import device

class MultiHeadAttentionLayer(Layer):
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        xp = device.xp
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = xp.random.randn(num_heads, d_model, self.d_k) * 0.01
        self.W_K = xp.random.randn(num_heads, d_model, self.d_k) * 0.01
        self.W_V = xp.random.randn(num_heads, d_model, self.d_k) * 0.01
        self.W_O = xp.random.randn(num_heads * self.d_k, d_model) * 0.01

        self.dW_Q = xp.zeros_like(self.W_Q)
        self.dW_K = xp.zeros_like(self.W_K)
        self.dW_V = xp.zeros_like(self.W_V)
        self.dW_O = xp.zeros_like(self.W_O)

        self.X = None
        self.heads = None
        self.attention_outputs = None

    def forward(self, X):
        xp = device.xp
        X = device.ensure_tensor(X)
        self.X = X
        batch, seq_len, _ = X.shape

        heads_out = []
        self.heads = []
        for i in range(self.num_heads):
            Q = X @ self.W_Q[i]
            K = X @ self.W_K[i]
            V = X @ self.W_V[i]

            scores = Q @ K.transpose(0,2,1) / math.sqrt(self.d_k)
            exp_scores = xp.exp(scores - xp.max(scores, axis=-1, keepdims=True))
            A = exp_scores / xp.sum(exp_scores, axis=-1, keepdims=True)

            out = A @ V
            heads_out.append(out)
            self.heads.append((Q, K, V, A))

        concat = xp.concatenate(heads_out, axis=-1)
        self.attention_outputs = concat

        out = concat @ self.W_O
        return out

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)

        self.dW_O = xp.tensordot(self.attention_outputs, dout, axes=([0,1],[0,1]))
        d_concat = dout @ self.W_O.T

        dX_total = xp.zeros_like(self.X)
        for i, (Q, K, V, A) in enumerate(self.heads):
            d_head = d_concat[..., i*self.d_k:(i+1)*self.d_k]

            d_scores = A * (d_head @ V.transpose(0,2,1) - xp.sum((d_head @ V.transpose(0,2,1)) * A, axis=-1, keepdims=True))
            dQ = d_scores @ K / math.sqrt(self.d_k)
            dK = d_scores.transpose(0,2,1) @ Q / math.sqrt(self.d_k)
            dV = A.transpose(0,2,1) @ d_head

            self.dW_Q[i] = xp.tensordot(self.X, dQ, axes=([0,1],[0,1]))
            self.dW_K[i] = xp.tensordot(self.X, dK, axes=([0,1],[0,1]))
            self.dW_V[i] = xp.tensordot(self.X, dV, axes=([0,1],[0,1]))

            dX_total += dQ @ self.W_Q[i].T + dK @ self.W_K[i].T + dV @ self.W_V[i].T

        return dX_total

    def parameters(self):
        return [self.W_Q, self.W_K, self.W_V, self.W_O]

    def gradients(self):
        return [self.dW_Q, self.dW_K, self.dW_V, self.dW_O]

__all__ = ["MultiHeadAttentionLayer"]
