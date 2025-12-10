from .base import Layer
from ..activations import Sigmoid, Tanh
from ..utils import device


class LSTMLayer(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        xp = device.xp
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        limit = (6.0 / (input_size + hidden_size)) ** 0.5
        
        self.W_fx = xp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_fh = xp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_f = xp.zeros(hidden_size, dtype=xp.float32)
        
        self.W_ix = xp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_ih = xp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_i = xp.zeros(hidden_size, dtype=xp.float32)
        
        self.W_ox = xp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_oh = xp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_o = xp.zeros(hidden_size, dtype=xp.float32)
        
        self.W_Cx = xp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_Ch = xp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_C = xp.zeros(hidden_size, dtype=xp.float32)
        
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.last_inputs = None
        self.last_hidden_states = None
        self.last_cell_states = None
        self.last_f = None
        self.last_i = None
        self.last_o = None
        self.last_C_tilde = None
        self.h0 = None
        self.C0 = None

        self.d_W_fx = xp.zeros_like(self.W_fx)
        self.d_W_fh = xp.zeros_like(self.W_fh)
        self.d_b_f = xp.zeros_like(self.b_f)

        self.d_W_ix = xp.zeros_like(self.W_ix)
        self.d_W_ih = xp.zeros_like(self.W_ih)
        self.d_b_i = xp.zeros_like(self.b_i)

        self.d_W_ox = xp.zeros_like(self.W_ox)
        self.d_W_oh = xp.zeros_like(self.W_oh)
        self.d_b_o = xp.zeros_like(self.b_o)

        self.d_W_Cx = xp.zeros_like(self.W_Cx)
        self.d_W_Ch = xp.zeros_like(self.W_Ch)
        self.d_b_C = xp.zeros_like(self.b_C)
    
    def forward(self, input_data):
        xp = device.xp
        input_data = device.ensure_tensor(input_data)
        batch_size, seq_len, _ = input_data.shape
        
        self.last_inputs = input_data
        hidden_states = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        cell_states = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        self.last_hidden_states = hidden_states
        self.last_cell_states = cell_states
        self.last_f = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        self.last_i = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        self.last_o = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        self.last_C_tilde = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        
        h = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        C = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        self.h0 = h
        self.C0 = C
        
        for t in range(seq_len):
            x_t = input_data[:, t, :]
            h_prev = h
            C_prev = C

            f_t = self.sigmoid.forward(x_t @ self.W_fx + h_prev @ self.W_fh + self.b_f)

            i_t = self.sigmoid.forward(x_t @ self.W_ix + h_prev @ self.W_ih + self.b_i)

            C_tilde = self.tanh.forward(x_t @ self.W_Cx + h_prev @ self.W_Ch + self.b_C)

            C = f_t * C_prev + i_t * C_tilde

            o_t = self.sigmoid.forward(x_t @ self.W_ox + h_prev @ self.W_oh + self.b_o)

            h = o_t * self.tanh.forward(C)
            
            hidden_states[:, t, :] = h
            cell_states[:, t, :] = C
            self.last_f[:, t, :] = f_t
            self.last_i[:, t, :] = i_t
            self.last_o[:, t, :] = o_t
            self.last_C_tilde[:, t, :] = C_tilde
        
        return hidden_states
    
    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)
        batch_size, seq_len, _ = dout.shape
        d_input = xp.zeros_like(self.last_inputs)
        
        dh_next = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        dC_next = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        
        for t in reversed(range(seq_len)):
            x_t = self.last_inputs[:, t, :]
            C_t = self.last_cell_states[:, t, :]
            h_prev = self.h0 if t == 0 else self.last_hidden_states[:, t - 1, :]
            C_prev = self.C0 if t == 0 else self.last_cell_states[:, t - 1, :]
            f_t = self.last_f[:, t, :]
            i_t = self.last_i[:, t, :]
            o_t = self.last_o[:, t, :]
            C_tilde = self.last_C_tilde[:, t, :]
            
            dh = dh_next + dout[:, t, :]

            dC = dC_next + (dh * o_t * self.tanh.backward(self.tanh.forward(C_t)))

            df = dC * C_prev
            di = dC * C_tilde
            dC_tilde = dC * i_t
            do = dh * self.tanh.forward(C_t)

            df_raw = df * self.sigmoid.backward(f_t)
            di_raw = di * self.sigmoid.backward(i_t)
            dC_tilde_raw = dC_tilde * self.tanh.backward(C_tilde)
            do_raw = do * self.sigmoid.backward(o_t)

            self.d_W_fx += x_t.T @ df_raw
            self.d_W_fh += h_prev.T @ df_raw
            self.d_b_f += xp.sum(df_raw, axis=0)
            
            self.d_W_ix += x_t.T @ di_raw
            self.d_W_ih += h_prev.T @ di_raw
            self.d_b_i += xp.sum(di_raw, axis=0)
            
            self.d_W_ox += x_t.T @ do_raw
            self.d_W_oh += h_prev.T @ do_raw
            self.d_b_o += xp.sum(do_raw, axis=0)
            
            self.d_W_Cx += x_t.T @ dC_tilde_raw
            self.d_W_Ch += h_prev.T @ dC_tilde_raw
            self.d_b_C += xp.sum(dC_tilde_raw, axis=0)

            d_input[:, t, :] = (
                df_raw @ self.W_fx.T +
                di_raw @ self.W_ix.T +
                do_raw @ self.W_ox.T +
                dC_tilde_raw @ self.W_Cx.T
            )

            dh_next = (
                df_raw @ self.W_fh.T +
                di_raw @ self.W_ih.T +
                do_raw @ self.W_oh.T +
                dC_tilde_raw @ self.W_Ch.T
            )
            dC_next = dC * f_t
        
        return d_input

    def parameters(self):
        return [self.W_fx, self.W_fh, self.b_f,
                self.W_ix, self.W_ih, self.b_i,
                self.W_ox, self.W_oh, self.b_o,
                self.W_Cx, self.W_Ch, self.b_C]

    def gradients(self):
        return [self.d_W_fx, self.d_W_fh, self.d_b_f,
                self.d_W_ix, self.d_W_ih, self.d_b_i,
                self.d_W_ox, self.d_W_oh, self.d_b_o,
                self.d_W_Cx, self.d_W_Ch, self.d_b_C]


__all__ = ["LSTMLayer"]

