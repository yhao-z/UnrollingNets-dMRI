import tensorflow as tf
import scipy.io as scio
from loguru import logger
from utils.tools import fft2c_mri, ifft2c_mri


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


# soft thresholding layer
class ST(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ST, self).__init__()
        self.GAP = tf.keras.layers.GlobalAveragePooling3D(keepdims=True) # batch_size, channels
        self.att = tf.keras.Sequential([
            tf.keras.layers.Dense(channels, activation='relu'),
            tf.keras.layers.Dense(channels, activation='sigmoid')
        ])
    
    def call(self, x, w):
        x_abs = tf.abs(x)
        gap = self.GAP(x_abs)
        attcoefs = self.att(gap)
        soft_thres = gap * attcoefs / w
        x_soft = tf.math.multiply(tf.math.sign(x), tf.nn.relu(x_abs - soft_thres))
        return x_soft
    

class JotlasNet(tf.keras.Model):
    def __init__(self, niter=15, channels=16):
        super(JotlasNet, self).__init__(name='JotlasNet')
        self.niter = niter
        self.channels = channels
        self.celllist = []

    def build(self, input_shape):
        for i in range(self.niter - 1):
            self.celllist.append(JotlasCell(input_shape, i, self.channels))
        self.celllist.append(JotlasCell(input_shape, self.niter - 1, self.channels, is_last=True))

    def call(self, d, F, mask):
        # nb, nc, nt, nx, ny = d.shape
        x_rec = F.TH(d)
        x_done = x_rec
        data = [x_rec, d, F, mask, x_done]

        for i in range(self.niter):
            data = self.celllist[i](data)

        x_rec = data[0]

        return x_rec, tf.constant(0.0)


class JotlasCell(tf.keras.layers.Layer):
    def __init__(self, input_shape, i, channels, is_last=False):
        super(JotlasCell, self).__init__()
        # self.nb, self.nt, self.nx, self.ny = input_shape

        self.thres_lr = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_lr %d' % i)
        self.mu = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='mu %d' % i)
        self.w = tf.Variable(tf.constant([0.5,0.5], dtype=tf.float32), trainable=True, name='w %d' % i)
        self.t = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='t %d' % i)

        self.sconv_1 = CNNLayer(n_f=channels, n_last=channels)
        self.sconv_2 = CNNLayer(n_f=channels, n_last=2)
        self.ST = ST(channels)

        self.lconv_1 = CNNLayer(n_f=channels, n_last=2)
        self.lconv_2 = CNNLayer(n_f=channels, n_last=2)
        
        self.i = i
        

    def call(self, data, **kwargs):
        x_rec, d, F, mask, x_done = data
        self.w = tf.nn.softmax(self.w)
        xg = self.gdecent_step(x_rec, d, F, mask)
        Z_tlr = self.prox_tlr(xg)
        Z_as = self.prox_as(xg)

        x = Z_tlr * tf.cast(self.w[0], tf.complex64) + Z_as * tf.cast(self.w[1], tf.complex64)
        x_rec = x + tf.cast(tf.sigmoid(self.t),tf.complex64) * (x - x_done)
        x_done = x
        data[0] = x_rec
        data[-1] = x_done

        return data


    def prox_tlr(self, x):
        [batch, Nt, Nx, Ny] = x.get_shape()
        x_in = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
        x = self.lconv_1(x_in)

        x_c = tf.complex(x[:, :, :, :, 0], x[:, :, :, :, 1])
        St, Ut, Vt = tf.linalg.svd(x_c)
        
        thres = tf.sigmoid(self.thres_lr) * St[..., 0] / self.w[0]
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

    def prox_as(self, x):
        [batch, Nt, Nx, Ny] = x.get_shape()
        x_in = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
        x = self.sconv_2(self.ST(self.sconv_1(x_in), self.w[1]))

        A = x + x_in
        A = tf.complex(A[:, :, :, :, 0], A[:, :, :, :, 1])

        return A

    def gdecent_step(self, x_pre, d, F, mask):
        k = F(x_pre)
        r = x_pre - tf.cast(tf.nn.relu(self.mu), tf.complex64) * F.TH(mask * k - d)
        return r
