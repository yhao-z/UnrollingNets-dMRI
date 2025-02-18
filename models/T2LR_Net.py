import tensorflow as tf

class CNNLayer(tf.keras.layers.Layer):
    def __init__(self, n_f=32, n_last=2, act_last=False):
        super(CNNLayer, self).__init__()
        self.mylayers = []

        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.ReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.ReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_last, 3, strides=1, padding='same', use_bias=False))
        if act_last:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        
        return res

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


class T2LR_Net(tf.keras.Model):
    def __init__(self, niter=15, channels=16):
        super(T2LR_Net, self).__init__(name='T2LR_Net')
        self.niter = niter
        self.channels = channels
        self.celllist = []

    def build(self, input_shape):
        for i in range(self.niter - 1):
            self.celllist.append(TLRCell(input_shape, i, self.channels))
        self.celllist.append(TLRCell(input_shape, self.niter - 1, self.channels, is_last=True))

    def call(self, d, F, mask):
        x_rec = F.TH(d)
        data = [x_rec, d, F, mask]

        for i in range(self.niter):
            data = self.celllist[i](data)

        x_rec = data[0]

        return x_rec, tf.constant(0.0)


class TLRCell(tf.keras.layers.Layer):
    def __init__(self, input_shape, i, channels, is_last=False):
        super(TLRCell, self).__init__()
        self.thres_lr = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_lr %d' % i)
        self.mu = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='mu %d' % i)

        self.lconv_1 = CNNLayer(n_f=channels, n_last=2)
        self.lconv_2 = CNNLayer(n_f=channels, n_last=2)
        

    def call(self, data, **kwargs):
        x_rec, d, F, mask = data
        xg = self.gdecent_step(x_rec, d, F, mask)
        x_rec = self.prox_tlr(xg)

        data[0] = x_rec

        return data


    def prox_tlr(self, x):
        [batch, Nt, Nx, Ny] = x.get_shape()
        x_in = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
        x = self.lconv_1(x_in)

        x_c = tf.complex(x[:, :, :, :, 0], x[:, :, :, :, 1])
        St, Ut, Vt = tf.linalg.svd(x_c)
        
        thres = tf.sigmoid(self.thres_lr) * St[..., 0]
        thres = tf.expand_dims(thres, -1)
        St = tf.nn.relu(St - thres)
        St = tf.linalg.diag(St)

        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0, 1, 3, 2])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        x_soft = tf.linalg.matmul(US, Vt_conj)

        x_soft = tf.stack([tf.math.real(x_soft), tf.math.imag(x_soft)], axis=-1)

        x_out = self.lconv_2(x_soft)

        A = x_out + x_in
        A = tf.complex(A[:, :, :, :, 0], A[:, :, :, :, 1])

        return A

    def gdecent_step(self, x_pre, d, F, mask):
        k = F(x_pre)
        r = x_pre - tf.cast(tf.nn.relu(self.mu), tf.complex64) * F.TH(mask * k - d)
        return r
