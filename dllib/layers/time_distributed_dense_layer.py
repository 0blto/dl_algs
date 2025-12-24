from dllib.layers import Layer, DenseLayer


class TimeDistributedDenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.dense = DenseLayer(input_size, output_size)

    def forward(self, X):
        batch, seq_len, dim = X.shape
        X_flat = X.reshape(-1, dim)
        out = self.dense.forward(X_flat)
        return out.reshape(batch, seq_len, -1)

    def backward(self, dout):
        batch, seq_len, dim = dout.shape
        dout_flat = dout.reshape(-1, dim)
        dx_flat = self.dense.backward(dout_flat)
        return dx_flat.reshape(batch, seq_len, -1)

    def parameters(self):
        return self.dense.parameters()

    def gradients(self):
        return self.dense.gradients()