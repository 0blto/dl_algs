from typing import Protocol, runtime_checkable


class TrainingModeMixin:
    def __init__(self):
        self.training = True

    def set_training_mode(self, mode: bool):
        self.training = mode

@runtime_checkable
class TrainableModel(Protocol):
    def forward(self, x): ...
    def backward(self, grad): ...
    def update_weights(self): ...
    def predict(self, x_test): ...
    def predict_proba(self, x_test): ...


__all__ = ["TrainingModeMixin", "TrainableModel"]