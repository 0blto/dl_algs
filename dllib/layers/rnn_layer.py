from .base import Layer
from ..activations import Tanh
from ..utils import device


class RNNLayer(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        xp = device.xp
        self.input_size = input_size
        self.hidden_size = hidden_size
        limit_xh = (6.0 / (input_size + hidden_size)) ** 0.5
        limit_hh = (6.0 / (hidden_size + hidden_size)) ** 0.5

        self.W_xh = xp.random.uniform(-limit_xh, limit_xh, (input_size, hidden_size)).astype(xp.float32)
        self.W_hh = xp.random.uniform(-limit_hh, limit_hh, (hidden_size, hidden_size)).astype(xp.float32)
        self.bias = xp.zeros(hidden_size, dtype=xp.float32)

        self.tanh = Tanh()

        self.last_inputs = None
        self.last_hidden_states = None
        self.last_pre_activations = None
        self.h0 = None

        self.dW_xh = None
        self.dW_hh = None
        self.dbias = None

    def forward(self, input_data):
        xp = device.xp
        input_data = device.ensure_tensor(input_data)
        batch_size, seq_len, _ = input_data.shape

        self.last_inputs = input_data
        hidden_states = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        pre_activations = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        self.last_hidden_states = hidden_states
        self.last_pre_activations = pre_activations

        h = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        self.h0 = h

        for t in range(seq_len):
            x_t = input_data[:, t, :]
            pre_act = x_t @ self.W_xh + h @ self.W_hh + self.bias
            h = self.tanh.forward(pre_act)
            hidden_states[:, t, :] = h
            pre_activations[:, t, :] = pre_act

        return hidden_states

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)
        batch_size, seq_len, _ = dout.shape

        dW_xh = xp.zeros_like(self.W_xh)
        dW_hh = xp.zeros_like(self.W_hh)
        dbias = xp.zeros_like(self.bias)
        d_input = xp.zeros_like(self.last_inputs)

        dh_next = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)

        for t in reversed(range(seq_len)):
            x_t = self.last_inputs[:, t, :]
            h_prev = self.h0 if t == 0 else self.last_hidden_states[:, t - 1, :]
            pre_act = self.last_pre_activations[:, t, :]

            dh = dh_next + dout[:, t, :]
            dh_raw = dh * self.tanh.backward(pre_act)

            dW_xh += x_t.T @ dh_raw
            dW_hh += h_prev.T @ dh_raw
            dbias += xp.sum(dh_raw, axis=0)

            d_input[:, t, :] = dh_raw @ self.W_xh.T
            dh_next = dh_raw @ self.W_hh.T

        self.dW_xh = dW_xh
        self.dW_hh = dW_hh
        self.dbias = dbias

        return d_input

    def parameters(self):
        return [self.W_xh, self.W_hh, self.bias]

    def gradients(self):
        return [self.dW_xh, self.dW_hh, self.dbias]


__all__ = ["RNNLayer"]

