from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def compute_metrics(self, y_pred, y_true) -> dict:
        pass


__all__ = ["Loss"]