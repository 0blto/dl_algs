from .base import Trainer
from ..losses.base import Loss
from ..utils import device
from ..utils.training import TrainableModel


class SimpleTrainer(Trainer):
    def __init__(self, loss: Loss):
        self.loss = loss

    def train(self, model: TrainableModel, x_train, y_train,
              x_test=None, y_test=None, batch_size=32, epochs=10, verbose=True):

        xp = device.xp
        n_samples = x_train.shape[0]

        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        for epoch in range(epochs):
            perm = xp.random.permutation(n_samples)
            x_train_shuffled = x_train[perm]
            y_train_shuffled = y_train[perm]

            train_loss_sum = 0.0
            train_acc_sum = 0.0
            acc_available = False

            for start in range(0, n_samples, batch_size):
                x_batch = x_train_shuffled[start:start + batch_size]
                y_batch = y_train_shuffled[start:start + batch_size]

                logits = model.forward(x_batch)

                self.loss.forward(logits, y_batch)
                grad = self.loss.backward()
                model.backward(grad)
                model.update_weights()

                metrics = self.loss.compute_metrics(logits, y_batch)
                train_loss_sum += metrics["loss"] * x_batch.shape[0]

                if metrics.get("accuracy") is not None:
                    acc_available = True
                    train_acc_sum += metrics["accuracy"] * x_batch.shape[0]

            avg_train_loss = train_loss_sum / n_samples
            avg_train_acc = train_acc_sum / n_samples if acc_available else None

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(avg_train_acc)

            if x_test is not None and y_test is not None:
                test_metrics = self.loss.compute_metrics(model.forward(x_test), y_test)
                history["test_loss"].append(test_metrics["loss"])
                history["test_acc"].append(test_metrics["accuracy"])
            else:
                history["test_loss"].append(None)
                history["test_acc"].append(None)

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f}"
                if avg_train_acc is not None: msg += f", Train Acc: {avg_train_acc:.5f}"
                msg += f" | Test Loss: {history['test_loss'][-1]:.5f}"
                if history['test_acc'][-1] is not None: msg += f", Test Acc: {history['test_acc'][-1]:.5f}"
                print(msg)

        return history

__all__ = ["SimpleTrainer"]