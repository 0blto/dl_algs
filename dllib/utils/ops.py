from dllib.utils import device


def softmax(logits):
    xp = device.xp
    logits = device.ensure_tensor(logits)

    shifted = logits - xp.max(logits, axis=1, keepdims=True)
    exp_y = xp.exp(shifted)
    probs = exp_y / xp.sum(exp_y, axis=1, keepdims=True)
    return probs

def one_hot_encode(labels, num_classes):
    xp = device.xp
    labels = device.ensure_tensor(labels).astype(xp.int32)
    one_hot = xp.zeros((labels.shape[0], num_classes), dtype=xp.float32)
    one_hot[xp.arange(labels.shape[0]), labels] = 1.0
    return one_hot


__all__ = ["softmax", "one_hot_encode"]