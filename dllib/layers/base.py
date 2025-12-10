from abc import ABC, abstractmethod



class Layer(ABC):
    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def parameters(self):
        return []

    @abstractmethod
    def gradients(self):
        return []


__all__ = ["Layer"]

