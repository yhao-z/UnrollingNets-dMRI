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


class ISTA_Net_plus(tf.keras.Model):
    def __init__(self, niter=10, channels=16):
        super(ISTA_Net_plus, self).__init__(name='ISTA_Net_plus')
        self.niter = niter
        self.channels = channels
        self.celllist = []

    def build(self, input_shape):
        for i in range(self.niter):
            self.celllist.append(ISTACell(input_shape, i, self.channels))

    def call(self, d, F, mask):
        # nb, nc, nt, nx, ny = d.shape
        x = F.TH(d)
        x_sym = tf.zeros_like(x)
        data = [x, x_sym, d, F, mask]

        X_SYM = []
        for i in range(self.niter):
            data = self.celllist[i](data)
            x_sym = data[1]
            X_SYM.append(x_sym)

        x_rec = data[0]

        return x_rec, X_SYM


class ISTACell(tf.keras.layers.Layer):
    def __init__(self, input_shape, i, channels):
        super(ISTACell, self).__init__()
        self.thres_coef = tf.Variable(tf.constant(0, dtype=tf.float32), trainable=True, name='thres_coef %d' % i)
        self.mu = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='mu %d' % i)

        self.conv_1 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_2 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_3 = CONV_OP(n_f=channels, ifactivate=False)

        self.conv_4 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_5 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_6 = CONV_OP(n_f=2, ifactivate=False)


    def call(self, data, **kwargs):
        x, x_sym, d, F, mask = data

        r = self.gdecent_step(x, d, F, mask)
        x, x_sym = self.prox_step(r)
        
        data[0] = x
        data[1] = x_sym

        return data

    def prox_step(self, r):
        [batch, Nt, Nx, Ny] = r.get_shape()
        x_in = tf.stack([tf.math.real(r), tf.math.imag(r)], axis=-1)
        x_1 = self.conv_1(x_in)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_soft = tf.math.multiply(tf.math.sign(x_3), tf.nn.relu(tf.abs(x_3) - tf.nn.relu(self.thres_coef)))

        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        A = x_6 + x_in
        A = tf.complex(A[:, :, :, :, 0], A[:, :, :, :, 1])

        x_1_sym = self.conv_4(x_3)
        x_1_sym = self.conv_5(x_1_sym)
        x_sym = x_1_sym - x_1 

        return A, x_sym


    def gdecent_step(self, x_pre, d, F, mask):
        k = F(x_pre)
        r = x_pre - tf.cast(tf.nn.relu(self.mu), tf.complex64) * F.TH(mask * k - d)
        return r
