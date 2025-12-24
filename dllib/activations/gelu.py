import math
from .base import Activation
from ..utils import device


class GELU(Activation):
    def forward(self, x):
        xp = device.xp
        x = device.ensure_tensor(x)

        return 0.5 * x * (
            1.0 + xp.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * xp.power(x, 3))
            )
        )

    def backward(self, z):
        xp = device.xp
        z = device.ensure_tensor(z)

        c = math.sqrt(2.0 / math.pi)
        z2 = xp.power(z, 2)
        z3 = z2 * z

        u = c * (z + 0.044715 * z3)
        t = xp.tanh(u)

        return 0.5 * (
            1.0
            + t
            + z * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * z2)
        )


__all__ = ["GELU"]