from ..utils import device


def col2im(cols, x_shape, field_h, field_w, padding, stride):
    xp = device.xp
    N, C, H, W = x_shape
    H_p, W_p = H + 2*padding, W + 2*padding

    x_padded = xp.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    out_h = (H_p - field_h) // stride + 1
    out_w = (W_p - field_w) // stride + 1

    cols = cols.reshape(C * field_h * field_w, N * out_h * out_w)
    cols = cols.reshape(C, field_h, field_w, N, out_h, out_w)
    cols = cols.transpose(3, 0, 4, 1, 5, 2)

    for y in range(out_h):
        ys = y * stride
        for x in range(out_w):
            xs = x * stride
            x_padded[:, :, ys:ys+field_h, xs:xs+field_w] += cols[:, :, y, :, x, :]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


def im2col(x, kernel, stride, padding):
    xp = device.xp

    if padding > 0:
        x = xp.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )

    N, C, H, W = x.shape
    out_h = (H - kernel) // stride + 1
    out_w = (W - kernel) // stride + 1

    i0 = xp.repeat(xp.arange(kernel), kernel)
    i1 = stride * xp.repeat(xp.arange(out_h), out_w)
    j0 = xp.tile(xp.arange(kernel), kernel)
    j1 = stride * xp.tile(xp.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    cols = x[:, :, i, j]
    cols = cols.transpose(1, 2, 0, 3).reshape(C * kernel * kernel, -1)

    return cols, out_h, out_w