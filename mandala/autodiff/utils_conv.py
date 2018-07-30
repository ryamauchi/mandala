from mandala import cuda


# modified from chainer.
def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def im2col(img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False,
           dy=1, dx=1, out_h=None, out_w=None):
    xp = cuda.get_array_module(img)
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)

    img = xp.pad(img,
                 ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                  mode='constant', constant_values=(pval,))
    col = xp.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    return col


def col2im(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    xp = cuda.get_array_module(col)

    n, c, kh, kw, out_h, out_w = col.shape
    img = xp.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                    dtype=col.dtype)
    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph:h + ph, pw:w + pw]
