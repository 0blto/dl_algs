from .base import Layer
from ..utils import device
from ..activations import Identity
from ..utils.conv import im2col, col2im


class ConvLayer(Layer):
    def __init__(self, num_filters, filter_size, input_depth,
                 padding=0, stride=1, activation=None):

        xp = device.xp
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_depth = input_depth
        self.padding = padding
        self.stride = stride
        self.d_filters = []
        self.d_biases = []
        self.activation = activation if activation is not None else Identity()

        fan_in = input_depth * filter_size * filter_size
        fan_out = num_filters * filter_size * filter_size
        limit = (6.0 / (fan_in + fan_out)) ** 0.5

        self.filters = xp.random.uniform(
            -limit, limit, (num_filters, input_depth, filter_size, filter_size)
        ).astype(xp.float32)

        self.biases = xp.zeros(num_filters, dtype=xp.float32)
        self.last_input = None
        self.last_x_cols = None
        self.last_output_pre_act = None

    def forward(self, x):
        x = device.ensure_tensor(x)
        self.last_input = x

        x_cols, out_h, out_w = im2col(
            x, self.filter_size, self.stride, self.padding
        )
        self.last_x_cols = x_cols

        w_cols = self.filters.reshape(self.num_filters, -1)

        out = w_cols @ x_cols
        out = out.reshape(self.num_filters, x.shape[0], out_h, out_w)
        out = out.transpose(1, 0, 2, 3)
        out = out + self.biases[None, :, None, None]

        self.last_output_pre_act = out
        return self.activation.forward(out)

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)

        d_act = self.activation.backward(self.last_output_pre_act)
        dout = dout * d_act

        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(self.num_filters, -1)

        d_filters = dout_reshaped @ self.last_x_cols.T
        self.d_filters = d_filters.reshape(self.filters.shape)

        self.d_biases = xp.sum(dout, axis=(0, 2, 3))

        w_cols = self.filters.reshape(self.num_filters, -1)
        dx_cols = w_cols.T @ dout_reshaped

        dx = col2im(
            dx_cols,
            self.last_input.shape,
            self.filter_size,
            self.filter_size,
            self.padding,
            self.stride,
        )

        return dx

    def parameters(self):
        return [self.filters, self.biases]

    def gradients(self):
        return [self.d_filters, self.d_biases]


__all__ = ["ConvLayer",]
