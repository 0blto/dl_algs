from dllib.layers import Layer
from dllib.utils import device


class MeanPoolingLayer(Layer):
    def forward(self, x):
        x = device.ensure_tensor(x)
        self.seq_len = x.shape[1]
        return x.mean(axis=1)

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)
        return xp.repeat(dout[:, None, :] / self.seq_len, self.seq_len, axis=1)

    def parameters(self): return []
    def gradients(self): return []


__all__ = ["MeanPoolingLayer"]