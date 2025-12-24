from dllib.layers import Layer, MultiHeadAttentionLayer, LayerNorm, TimeDistributedDenseLayer
from dllib.layers.activation_layers import GELULayer


class TransformerLayer(Layer):
    def __init__(self, d_model, num_heads, ff_hidden_dim):
        self.mha = MultiHeadAttentionLayer(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = [
            (TimeDistributedDenseLayer(d_model, ff_hidden_dim), GELULayer()),
            (TimeDistributedDenseLayer(ff_hidden_dim, d_model), GELULayer())
        ]
        self.norm2 = LayerNorm(d_model)

    def forward(self, X):
        X = self.norm1.forward(X + self.mha.forward(X))
        ff_out = self.ff[0][1].forward(self.ff[0][0].forward(X))
        ff_out = self.ff[1][1].forward(self.ff[1][0].forward(ff_out))
        return self.norm2.forward(X + ff_out)

    def backward(self, dout):
        dX = self.norm2.backward(dout)
        dff = self.ff[1][0].backward(self.ff[1][1].backward(dX))
        dff = self.ff[0][0].backward(self.ff[0][1].backward(dff))
        dX = dX + dff
        dX = self.norm1.backward(dX)
        dX += self.mha.backward(dX)
        return dX

    def parameters(self):
        params = self.mha.parameters()
        params += self.ff[0][0].parameters() + self.ff[0][1].parameters()
        params += self.ff[1][0].parameters() + self.ff[1][1].parameters()
        params += self.norm1.parameters() + self.norm2.parameters()
        return params

    def gradients(self):
        grads = self.mha.gradients()
        grads += self.ff[0][0].gradients() + self.ff[0][1].gradients()
        grads += self.ff[1][0].gradients() + self.ff[1][1].gradients()
        grads += self.norm1.gradients() + self.norm2.gradients()
        return grads


__all__ = ["TransformerLayer"]