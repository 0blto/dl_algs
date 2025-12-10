from .base import Layer
from ..activations import Sigmoid, Tanh
from ..utils import device


class GRULayer(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        xp = device.xp
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        limit = (6.0 / (input_size + hidden_size)) ** 0.5
        
        self.W_rx = xp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_rh = xp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_r = xp.zeros(hidden_size, dtype=xp.float32)
        
        self.W_zx = xp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_zh = xp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_z = xp.zeros(hidden_size, dtype=xp.float32)
        
        self.W_hx = xp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_hh = xp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_h = xp.zeros(hidden_size, dtype=xp.float32)
        
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.last_inputs = None
        self.last_hidden_states = None
        self.last_r = None
        self.last_z = None
        self.last_h_tilde = None
        self.h0 = None

        self.d_W_rx = xp.zeros_like(self.W_rx)
        self.d_W_rh = xp.zeros_like(self.W_rh)
        self.d_b_r = xp.zeros_like(self.b_r)

        self.d_W_zx = xp.zeros_like(self.W_zx)
        self.d_W_zh = xp.zeros_like(self.W_zh)
        self.d_b_z = xp.zeros_like(self.b_z)

        self.d_W_hx = xp.zeros_like(self.W_hx)
        self.d_W_hh = xp.zeros_like(self.W_hh)
        self.d_b_h = xp.zeros_like(self.b_h)
    
    def forward(self, input_data):
        xp = device.xp
        input_data = device.ensure_tensor(input_data)
        batch_size, seq_len, _ = input_data.shape
        
        self.last_inputs = input_data
        hidden_states = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        self.last_hidden_states = hidden_states
        self.last_r = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        self.last_z = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        self.last_h_tilde = xp.zeros((batch_size, seq_len, self.hidden_size), dtype=xp.float32)
        
        h = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        self.h0 = h
        
        for t in range(seq_len):
            x_t = input_data[:, t, :]
            h_prev = h
            
            r_t = self.sigmoid.forward(x_t @ self.W_rx + h_prev @ self.W_rh + self.b_r)
            
            z_t = self.sigmoid.forward(x_t @ self.W_zx + h_prev @ self.W_zh + self.b_z)
            
            h_tilde = self.tanh.forward(x_t @ self.W_hx + (r_t * h_prev) @ self.W_hh + self.b_h)
            
            h = (1 - z_t) * h_prev + z_t * h_tilde
            
            hidden_states[:, t, :] = h
            self.last_r[:, t, :] = r_t
            self.last_z[:, t, :] = z_t
            self.last_h_tilde[:, t, :] = h_tilde
        
        return hidden_states
    
    def backward(self, dout):
        xp = device.xp
        dout = device.ensure_tensor(dout)
        batch_size, seq_len, _ = dout.shape
        d_input = xp.zeros_like(self.last_inputs)
        
        dh_next = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        
        for t in reversed(range(seq_len)):
            x_t = self.last_inputs[:, t, :]
            h_prev = self.h0 if t == 0 else self.last_hidden_states[:, t - 1, :]
            r_t = self.last_r[:, t, :]
            z_t = self.last_z[:, t, :]
            h_tilde = self.last_h_tilde[:, t, :]
            
            dh = dh_next + dout[:, t, :]

            dh_tilde = dh * z_t
            dz = dh * (h_tilde - h_prev)

            dh_tilde_raw = dh_tilde * self.tanh.backward(h_tilde)

            dr = (dh_tilde_raw @ self.W_hh.T) * h_prev
            dr_raw = dr * self.sigmoid.backward(r_t)

            dz_raw = dz * self.sigmoid.backward(z_t)

            self.d_W_hx += x_t.T @ dh_tilde_raw
            self.d_W_hh += (r_t * h_prev).T @ dh_tilde_raw
            self.d_b_h += xp.sum(dh_tilde_raw, axis=0)
            
            self.d_W_rx += x_t.T @ dr_raw
            self.d_W_rh += h_prev.T @ dr_raw
            self.d_b_r += xp.sum(dr_raw, axis=0)
            
            self.d_W_zx += x_t.T @ dz_raw
            self.d_W_zh += h_prev.T @ dz_raw
            self.d_b_z += xp.sum(dz_raw, axis=0)

            d_input[:, t, :] = (
                dh_tilde_raw @ self.W_hx.T +
                dr_raw @ self.W_rx.T +
                dz_raw @ self.W_zx.T
            )

            dh_next = (
                dh * (1 - z_t) +
                (dh_tilde_raw @ self.W_hh.T) * r_t +
                dr_raw @ self.W_rh.T +
                dz_raw @ self.W_zh.T
            )
        
        return d_input

    def parameters(self):
        return [self.W_rx, self.W_rh, self.b_r,
                self.W_zx, self.W_zh, self.b_z,
                self.W_hx, self.W_hh, self.b_h]

    def gradients(self):
        return [self.d_W_rx, self.d_W_rh, self.d_b_r,
                self.d_W_zx, self.d_W_zh, self.d_b_z,
                self.d_W_hx, self.d_W_hh, self.d_b_h]


__all__ = ["GRULayer"]

