from typing import Optional

from .base import Loss
from ..utils import device, softmax, one_hot_encode


class CrossEntropyLoss(Loss):
    def __init__(self, num_classes: Optional[int] = None, eps: float = 1e-12):
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)

        probs = softmax(y_pred)
        one_hot = one_hot_encode(y_true, self.num_classes or probs.shape[1])
        loss = -xp.sum(one_hot * xp.log(xp.clip(probs, self.eps, 1 - self.eps))) / y_pred.shape[0]
        return float(device.to_cpu(loss))

    def backward(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)

        probs = softmax(y_pred)
        one_hot = one_hot_encode(y_true, self.num_classes or probs.shape[1])
        return (probs - one_hot) / y_pred.shape[0]

    def compute_metrics(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)

        probs = softmax(y_pred)
        preds = xp.argmax(probs, axis=1)
        accuracy = float(device.to_cpu(xp.sum(preds == y_true))) / y_true.shape[0]

        one_hot = one_hot_encode(y_true, probs.shape[1])
        loss = -xp.sum(one_hot * xp.log(xp.clip(probs, self.eps, 1 - self.eps))) / y_true.shape[0]
        return {"loss": float(device.to_cpu(loss)), "accuracy": accuracy}


__all__ = ["CrossEntropyLoss"]

