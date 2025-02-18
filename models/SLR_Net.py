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

class SLR_Net(tf.keras.Model):
    def __init__(self, niter=10, channels=16):
        super(SLR_Net, self).__init__(name='SLR_Net')
        self.niter = niter
        self.channels = channels
        self.celllist = []
    

    def build(self, input_shape):
        for i in range(self.niter-1):
            self.celllist.append(SLRCell(input_shape, self.channels))
        self.celllist.append(SLRCell(input_shape, self.channels, is_last=True))

    def call(self, d, F, mask):
        X_SYM = []
        x_rec = F.TH(d)
        t = tf.zeros_like(x_rec)
        beta = tf.zeros_like(x_rec)
        x_sym = tf.zeros_like(x_rec)
        data = [x_rec, x_sym, beta, t, d, F, mask]
        
        for i in range(self.niter):
            data = self.celllist[i](data, d.shape)
            x_sym = data[1]
            X_SYM.append(x_sym)

        x_rec = data[0]
        
        return x_rec, X_SYM


class SLRCell(tf.keras.layers.Layer):
    def __init__(self, input_shape, channels, is_last=False):
        super(SLRCell, self).__init__()
        if is_last:
            self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=False, name='thres_coef')
            self.eta = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=False, name='eta')
        else:
            self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef')
            self.eta = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=True, name='eta')

        self.conv_1 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_2 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_3 = CONV_OP(n_f=channels, ifactivate=False)
        self.conv_4 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_5 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_6 = CONV_OP(n_f=2, ifactivate=False)

        self.lambda_step = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda_1')
        self.lambda_step_2 = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda_2')
        self.soft_thr = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='soft_thr')


    def call(self, data, input_shape):
        # self.nb, self.nc, self.nt, self.nx, self.ny = input_shape
        x_rec, x_sym, beta, t, d, F, mask = data

        x_rec, x_sym = self.sparse(x_rec, d, t, beta, F, mask)
        t = self.lowrank(x_rec)
   
        beta = self.beta_mid(beta, x_rec, t)

        data[0] = x_rec
        data[1] = x_sym
        data[2] = beta
        data[3] = t

        return data

    def sparse(self, x_rec, d, t, beta, F, mask):
        lambda_step = tf.cast(tf.nn.relu(self.lambda_step), tf.complex64)
        lambda_step_2 = tf.cast(tf.nn.relu(self.lambda_step_2), tf.complex64)

        ATAX_cplx = F.TH(mask * F(x_rec) - d)

        r_n = x_rec - tf.math.scalar_mul(lambda_step, ATAX_cplx) +\
              tf.math.scalar_mul(lambda_step_2, x_rec + beta - t)

        # D_T(soft(D_r_n))
        if len(r_n.shape) == 4:
            r_n = tf.stack([tf.math.real(r_n), tf.math.imag(r_n)], axis=-1)

        x_1 = self.conv_1(r_n)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_soft = tf.math.multiply(tf.math.sign(x_3), tf.nn.relu(tf.abs(x_3) - self.soft_thr))

        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        x_rec = x_6 + r_n

        x_1_sym = self.conv_4(x_3)
        x_1_sym = self.conv_5(x_1_sym)
        x_1_sym = self.conv_6(x_1_sym)

        x_sym = x_1_sym - r_n
        x_rec = tf.complex(x_rec[:, :, :, :, 0], x_rec[:, :, :, :, 1])

        return x_rec, x_sym

    def lowrank(self, x_rec):
        [batch, Nt, Nx, Ny] = x_rec.get_shape()
        M = tf.reshape(x_rec, [batch, Nt, Nx*Ny])
        St, Ut, Vt = tf.linalg.svd(M)

        thres = tf.sigmoid(self.thres_coef) * St[:, 0]
        thres = tf.expand_dims(thres, -1)
        St = tf.nn.relu(St - thres)

        St = tf.linalg.diag(St)
        
        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0, 2, 1])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        M = tf.linalg.matmul(US, Vt_conj)
        x_rec = tf.reshape(M, [batch, Nt, Nx, Ny])

        return x_rec

    def beta_mid(self, beta, x_rec, t):
        eta = tf.cast(tf.nn.relu(self.eta), tf.complex64)
        return beta + tf.multiply(eta, x_rec - t)
