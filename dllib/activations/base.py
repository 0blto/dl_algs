from abc import ABC, abstractmethod

from ..utils import device


class Activation(ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, output):
        pass


class Identity(Activation):
    def forward(self, x):
        return device.ensure_tensor(x)
    
    def backward(self, output):
        xp = device.xp
        return xp.ones_like(output)


__all__ = ["Activation", "Identity"]

