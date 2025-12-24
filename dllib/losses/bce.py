from .base import Loss
from ..utils import device

class BCELoss(Loss):
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)

        y_pred_clipped = xp.clip(y_pred, self.eps, 1 - self.eps)
        loss = -xp.mean(y_true * xp.log(y_pred_clipped) + (1 - y_true) * xp.log(1 - y_pred_clipped))
        return float(device.to_cpu(loss))

    def backward(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)
        y_pred_clipped = xp.clip(y_pred, self.eps, 1 - self.eps)
        return (y_pred_clipped - y_true) / y_pred.shape[0]

    def compute_metrics(self, y_pred, y_true) -> dict:
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)
        preds = (y_pred > 0.5).astype(y_true.dtype)
        accuracy = float(device.to_cpu(xp.sum(preds == y_true))) / y_true.size
        loss = self.forward(y_pred, y_true)
        return {"loss": loss, "accuracy": accuracy}



__all__ = ["BCELoss"]
