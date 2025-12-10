import numpy as np

xp = np
USE_GPU = False


def configure_backend(array_module, use_gpu: bool) -> None:
    global xp, USE_GPU
    xp = array_module
    USE_GPU = use_gpu


def ensure_tensor(array):
    return array if isinstance(array, xp.ndarray if USE_GPU else np.ndarray) else xp.asarray(array)


def to_cpu(array):
    return xp.asnumpy(array) if USE_GPU and hasattr(xp, "asnumpy") else array


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    xp.random.seed(seed)


__all__ = [
    "configure_backend",
    "ensure_tensor",
    "seed_everything",
    "to_cpu",
    "xp",
    "USE_GPU",
]

