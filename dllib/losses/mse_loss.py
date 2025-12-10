from dllib.losses.base import Loss
from dllib.utils import device


class MSELoss(Loss):
    def __init__(self):
        self.diff = None

    def forward(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)
        if y_pred.shape != y_true.shape: y_true = y_true.reshape(y_pred.shape)
        self.diff = y_pred - y_true
        loss = (self.diff ** 2).mean()
        return float(device.to_cpu(loss))

    def backward(self):
        return 2 * self.diff / self.diff.size

    def compute_metrics(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)
        if y_pred.shape != y_true.shape: y_true = y_true.reshape(y_pred.shape)
        diff = y_pred - y_true
        loss = (diff ** 2).mean()
        return {"loss": float(device.to_cpu(loss)), "accuracy": None}


__all__ = ["MSELoss"]
