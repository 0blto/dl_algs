from typing import Tuple

import numpy as np


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    n_samples = len(features)
    
    for i in range(n_samples - sequence_length):
        X_seq.append(features[i:i + sequence_length])
        y_seq.append(target[i + sequence_length])
    
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


__all__ = ["create_sequences"]

