from abc import ABC, abstractmethod

from ..utils.training import TrainableModel


class Trainer(ABC):
    @abstractmethod
    def train(self, model: TrainableModel, x_train, y_train, x_test=None, y_test=None, epochs=10):
        pass


__all__ = ["Trainer"]
