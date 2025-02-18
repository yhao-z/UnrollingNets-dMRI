import tensorflow as tf

class CONV_OP(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate == True:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res


class DCCNN(tf.keras.Model):
    def __init__(self, niter=10, channels=16):
        super(DCCNN, self).__init__(name='DCCNN')
        self.niter = niter
        self.channels = channels
        self.celllist = []

    def build(self, input_shape):
        for _ in range(self.niter):
            self.celllist.append(DCell(input_shape, self.channels))

    def call(self, d, F, mask):
        # nb, nc, nt, nx, ny = d.shape
        x_rec = F.TH(d)
        data = [x_rec, d, F, mask]

        for i in range(self.niter):
            data = self.celllist[i](data)

        x_rec = data[0]

        return x_rec


class DCell(tf.keras.layers.Layer):
    def __init__(self, input_shape, channels):
        super(DCell, self).__init__()

        self.conv_1 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_2 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_3 = CONV_OP(n_f=channels, ifactivate=True)

        self.conv_4 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_5 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_6 = CONV_OP(n_f=2, ifactivate=False)

    def call(self, data, **kwargs):
        x_rec, d, F, mask = data

        A = self.prox_step(x_rec)
        x_rec = self.dc_step(A, d, F, mask)

        data[0] = x_rec

        return data

    def prox_step(self, x):
        [batch, Nt, Nx, Ny] = x.get_shape()
        x_in = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
        x_1 = self.conv_1(x_in)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
        x_4 = self.conv_4(x_3)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        A = x_6 + x_in
        A = tf.complex(A[:, :, :, :, 0], A[:, :, :, :, 1])

        return A

    def dc_step(self, A, d, F, mask):
        k = F(A)
        k_rec = (1 - mask) * k + d
        x_rec = F.TH(k_rec)
        return x_rec