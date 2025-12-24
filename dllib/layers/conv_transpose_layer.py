from .base import Layer
from ..utils import device
from ..utils.conv import im2col, col2im


class ConvTransposeLayer(Layer):
    def __init__(self, input_depth, num_filters, filter_size,
                 padding=0, stride=1):

        xp = device.xp
        self.input_depth = input_depth
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride

        self.d_filters = []
        self.d_biases = []

        fan_in = input_depth * filter_size * filter_size
        fan_out = num_filters * filter_size * filter_size
        limit = (6.0 / (fan_in + fan_out)) ** 0.5

        self.filters = xp.random.uniform(
            -limit, limit,
            (input_depth, num_filters, filter_size, filter_size)
        ).astype(xp.float32)

        self.biases = xp.zeros(num_filters, dtype=xp.float32)

        self.last_input = None
        self.last_output = None

    def forward(self, x):
        x = device.ensure_tensor(x)
        self.last_input = x

        batch, _, h, w = x.shape
        out_h = (h - 1) * self.stride - 2 * self.padding + self.filter_size
        out_w = (w - 1) * self.stride - 2 * self.padding + self.filter_size

        x_cols = x.transpose(1, 0, 2, 3).reshape(self.input_depth, -1)
        w_cols = self.filters.reshape(self.input_depth, -1)

        out_cols = w_cols.T @ x_cols
        out = col2im(
            out_cols,
            (batch, self.num_filters, out_h, out_w),
            self.filter_size,
            self.filter_size,
            self.padding,
            self.stride,
        )

        out = out + self.biases[None, :, None, None]
        self.last_output = out

        return out

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)

        dout_cols, _, _ = im2col(
            dout, self.filter_size, self.stride, self.padding
        )

        x_cols = self.last_input.transpose(1, 0, 2, 3).reshape(self.input_depth, -1)

        self.d_filters = (x_cols @ dout_cols.T).reshape(self.filters.shape)
        self.d_biases = xp.sum(dout, axis=(0, 2, 3))

        w_cols = self.filters.reshape(self.input_depth, -1)
        dx_cols = w_cols @ dout_cols

        dx = dx_cols.reshape(self.last_input.shape)
        return dx

    def parameters(self):
        return [self.filters, self.biases]

    def gradients(self):
        return [self.d_filters, self.d_biases]


__all__ = ["ConvTransposeLayer"]
