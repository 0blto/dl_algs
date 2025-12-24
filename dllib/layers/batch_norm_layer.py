from .base import Layer
from ..utils import device, TrainingModeMixin


class BatchNormLayer(Layer, TrainingModeMixin):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        xp = device.xp

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = xp.ones(num_features, dtype=xp.float32)
        self.beta = xp.zeros(num_features, dtype=xp.float32)

        self.running_mean = xp.zeros(num_features, dtype=xp.float32)
        self.running_var = xp.ones(num_features, dtype=xp.float32)

        self.d_gamma = None
        self.d_beta = None

        # cache
        self.x_centered = None
        self.std_inv = None
        self.x_norm = None

    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)

        if x.ndim == 4:
            axes = (0, 2, 3)
            reshape_shape = (1, -1, 1, 1)
        else:
            axes = (0,)
            reshape_shape = (1, -1)

        if self.training:
            mean = xp.mean(x, axis=axes)
            var = xp.var(x, axis=axes)

            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        self.x_centered = x - mean.reshape(reshape_shape)
        self.std_inv = 1.0 / xp.sqrt(var + self.eps)
        self.x_norm = self.x_centered * self.std_inv.reshape(reshape_shape)

        out = (
            self.gamma.reshape(reshape_shape) * self.x_norm
            + self.beta.reshape(reshape_shape)
        )
        return out

    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)

        if dout.ndim == 4:
            axes = (0, 2, 3)
            reshape_shape = (1, -1, 1, 1)
            N = dout.shape[0] * dout.shape[2] * dout.shape[3]
        else:
            axes = (0,)
            reshape_shape = (1, -1)
            N = dout.shape[0]

        self.d_gamma = xp.sum(dout * self.x_norm, axis=axes)
        self.d_beta = xp.sum(dout, axis=axes)

        gamma = self.gamma.reshape(reshape_shape)
        std_inv = self.std_inv.reshape(reshape_shape)
        x_centered = self.x_centered

        dx_norm = dout * gamma

        dvar = xp.sum(
            dx_norm * x_centered * -0.5 * std_inv ** 3,
            axis=axes
        )

        dmean = (
                xp.sum(dx_norm * -std_inv, axis=axes)
                + dvar * xp.mean(-2.0 * x_centered, axis=axes)
        )

        dx = (
                dx_norm * std_inv
                + dvar.reshape(reshape_shape) * 2 * x_centered / N
                + dmean.reshape(reshape_shape) / N
        )

        return dx

    def parameters(self):
        return [self.gamma, self.beta]

    def gradients(self):
        return [self.d_gamma, self.d_beta]


__all__ = ["BatchNormLayer"]