from typing import Optional

from .base import Loss
from ..utils import device, softmax, one_hot_encode


class CrossEntropyLoss(Loss):
    def __init__(self, num_classes: Optional[int] = None):
        self.num_classes = num_classes
        self.probs = None
        self.one_hot = None
        self.eps = 1e-12

    def forward(self, y_pred, y_true):
        xp = device.xp
        self.probs = softmax(y_pred)
        self.one_hot = one_hot_encode(y_true, self.num_classes)
        loss = -xp.sum(self.one_hot * xp.log(xp.clip(self.probs, self.eps, 1 - self.eps))) / y_pred.shape[0]
        return float(device.to_cpu(loss))

    def backward(self): return (self.probs - self.one_hot) / self.probs.shape[0]

    def compute_metrics(self, y_pred, y_true):
        xp = device.xp
        probs = softmax(y_pred)
        preds = xp.argmax(probs, axis=1)
        accuracy = float(device.to_cpu(xp.sum(preds == y_true))) / y_true.shape[0]
        one_hot = one_hot_encode(y_true, probs.shape[1])
        eps = 1e-12
        loss = -xp.sum(one_hot * xp.log(xp.clip(probs, eps, 1 - eps))) / y_true.shape[0]
        return {
            "loss": float(device.to_cpu(loss)),
            "accuracy": accuracy
        }


__all__ = ["CrossEntropyLoss"]

