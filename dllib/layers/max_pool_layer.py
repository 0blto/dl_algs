from .base import Layer
from ..utils import device


class MaxPoolLayer(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None
        self.max_indices = None

    def forward(self, input_data):
        xp = device.xp
        x = device.ensure_tensor(input_data)

        N, C, H, W = x.shape

        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        h0 = xp.arange(self.pool_size).reshape(-1, 1)
        h1 = self.stride * xp.arange(out_h).reshape(1, -1)
        h_idx = h0 + h1

        w0 = xp.arange(self.pool_size).reshape(-1, 1)
        w1 = self.stride * xp.arange(out_w).reshape(1, -1)
        w_idx = w0 + w1

        windows = x[:, :, h_idx[:, :, None], w_idx[:, None, :]]

        windows_reshaped = windows.reshape(N, C, -1, out_h, out_w)

        max_values = windows_reshaped.max(axis=2)
        max_idx_flat = windows_reshaped.argmax(axis=2)

        h_off = max_idx_flat // self.pool_size
        w_off = max_idx_flat % self.pool_size

        self.max_indices = xp.zeros((N, C, out_h, out_w, 2), dtype=xp.int32)
        self.max_indices[..., 0] = h_idx[h_off, xp.arange(out_h)[None, :, None]]
        self.max_indices[..., 1] = w_idx[w_off, xp.arange(out_w)[None, None, :]]

        self.last_input = x
        return max_values

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)
        N, C, H, W = self.last_input.shape
        _, _, out_h, out_w = dout.shape

        d_input = xp.zeros_like(self.last_input)

        max_h = self.max_indices[..., 0]
        max_w = self.max_indices[..., 1]
        n_idx = xp.arange(N)[:, None, None, None]
        c_idx = xp.arange(C)[None, :, None, None]
        n_idx = xp.broadcast_to(n_idx, (N, C, out_h, out_w))
        c_idx = xp.broadcast_to(c_idx, (N, C, out_h, out_w))
        flat_idx = xp.ravel_multi_index(
            (n_idx.ravel(), c_idx.ravel(), max_h.ravel(), max_w.ravel()),
            dims=d_input.shape
        )
        xp.add.at(d_input.ravel(), flat_idx, dout.ravel())

        return d_input

    def parameters(self):
        return []

    def gradients(self):
        return []


__all__ = ["MaxPoolLayer"]

