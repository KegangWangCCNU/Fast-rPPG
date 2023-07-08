from weights import *

weights = [np.array(i) for i in weights]

def N(x, ax=2):
    return (x-x.mean(axis=ax, keepdims=True))/(x.std(axis=ax, keepdims=True)+1e-6)

def conv3d_pad(inputs, filters, strides=[1, 1, 1], padding="SAME"):
    B, C_in, D, H, W = inputs.shape
    C_out, C_in, K_d, K_h, K_w = filters.shape

    if padding == "VALID":
        D_out = int(np.ceil((D - K_d + 1) / strides[0]))
        H_out = int(np.ceil((H - K_h + 1) / strides[1]))
        W_out = int(np.ceil((W - K_w + 1) / strides[2]))
    else:
        D_out = int(D / strides[0])
        H_out = int(H / strides[1])
        W_out = int(W / strides[2])

    if padding == "SAME":
        pad_d = (D_out - 1) * strides[0] + K_d - D
        pad_h = (H_out - 1) * strides[1] + K_h - H
        pad_w = (W_out - 1) * strides[2] + K_w - W
        pad_top = int(pad_d / 2)
        pad_bottom = pad_d - pad_top
        pad_left = int(pad_h / 2)
        pad_right = pad_h - pad_left
        pad_front = int(pad_w / 2)
        pad_back = pad_w - pad_front
        inputs = np.pad(inputs, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'edge')

    outputs = np.zeros((B, C_out, D_out, H_out, W_out))

    return conv3d(outputs, B, C_out, C_in, D, H, W, K_d, K_h, K_w, strides, inputs, filters)

def conv3d(outputs, B, C_out, C_in, D, H, W, K_d, K_h, K_w, strides, inputs, filters):
    """
    Using numba.jit can reduce the latency to around 1ms.

    import numba
    conv3d = numba.jit(conv3d)
    """
    for b in range(B):
        for c_out in range(C_out):
            for c_in in range(C_in):
                for d in range(0, D - K_d + 1, strides[0]):
                    for h in range(0, H - K_h + 1, strides[1]):
                        for w in range(0, W - K_w + 1, strides[2]):
                            cur_input = inputs[b, c_in, d:d + K_d, h:h + K_h, w:w + K_w]
                            cur_filter = filters[c_out, c_in]
                            cur_output = np.sum(cur_input * cur_filter)
                            outputs[b, c_out, d // strides[0], h // strides[1], w // strides[2]] += cur_output
    return outputs

def model(x):
    x = N(x.transpose(0, 4, 1, 2, 3))
    x = N(np.tanh(conv3d_pad(x, weights[0], (1, 2, 2), 'SAME')+weights[1]))
    x = N(np.tanh(conv3d_pad(x, weights[2], (1, 2, 2), 'SAME')+weights[3]))
    x = conv3d_pad(x, np.ones((2, 2, 1, 2, 2)), (1, 2, 2), 'VALID')
    x = conv3d_pad(x, weights[4], (1, 1, 1), 'SAME')+weights[5]
    return x.reshape(x.shape[0], -1)