from dllib.losses.base import Loss
from dllib.utils import device


class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)
        if y_pred.shape != y_true.shape:
            y_true = y_true.reshape(y_pred.shape)
        loss = xp.mean((y_pred - y_true) ** 2)
        return float(device.to_cpu(loss))

    def backward(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)
        if y_pred.shape != y_true.shape:
            y_true = y_true.reshape(y_pred.shape)
        return 2 * (y_pred - y_true) / y_pred.size

    def compute_metrics(self, y_pred, y_true):
        xp = device.xp
        y_pred = device.ensure_tensor(y_pred)
        y_true = device.ensure_tensor(y_true)
        if y_pred.shape != y_true.shape:
            y_true = y_true.reshape(y_pred.shape)
        loss = xp.mean((y_pred - y_true) ** 2)
        return {"loss": float(device.to_cpu(loss)), "accuracy": None}


__all__ = ["MSELoss"]
