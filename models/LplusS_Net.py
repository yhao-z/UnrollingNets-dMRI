import tensorflow as tf


class CNNLayer(tf.keras.layers.Layer):
    def __init__(self, n_f=32):
        super(CNNLayer, self).__init__()
        self.mylayers = []

        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(2, 3, strides=1, padding='same', use_bias=False))
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        if len(input.shape) == 4:
            input2c = tf.stack([tf.math.real(input), tf.math.imag(input)], axis=-1)
        else:
            input2c = tf.concat([tf.math.real(input), tf.math.imag(input)], axis=-1)
        res = self.seq(input2c)
        res = tf.complex(res[:,:,:,:,0], res[:,:,:,:,1])
        
        return res


class LplusS_Net(tf.keras.Model):
    def __init__(self, niter=10, channels=32):
        super(LplusS_Net, self).__init__(name='LplusS_Net')
        self.niter = niter
        self.channels = channels

        self.celllist = []
    
    def build(self, input_shape):
        for i in range(self.niter-1):
            self.celllist.append(LSCell_learned_step(input_shape, self.channels, is_last=False))
        self.celllist.append(LSCell_learned_step(input_shape, self.channels, is_last=True))

    def call(self, d, F, mask):
        if len(d.shape) == 4:
            nb, nt, nx, ny = d.shape
        elif len(d.shape) == 5:
            nb, nc, nt, nx, ny = d.shape
        Lpre = tf.reshape(F.TH(d), [nb, nt, nx*ny])
        Spre = tf.zeros_like(Lpre)
        Mpre = Lpre

        data = [Lpre, Spre, Mpre, d, F, mask]

        for i in range(self.niter):
            data = self.celllist[i](data, d.shape)
        
        L, S, M, *_ = data
        M = tf.reshape(M, [nb, nt, nx, ny])
        L = tf.reshape(L, [nb, nt, nx, ny])
        S = tf.reshape(S, [nb, nt, nx, ny])

        return L, S, M

class LSCell_learned_step(tf.keras.layers.Layer):
    def __init__(self, input_shape, channels, is_last):
        super(LSCell_learned_step, self).__init__()
        self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True)
        self.sconv = CNNLayer(n_f=channels)
        self.is_last = is_last
        if not is_last:
            self.gamma = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=True)

    def call(self, data, input_shape):
        if len(input_shape)==4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        elif len(input_shape)==5:
            self.nb, self.nc, self.nt, self.nx, self.ny = input_shape

        Lpre, Spre, Mpre, d, F, mask = data

        L = self.lowrank(Mpre)
        S = self.sparse(Mpre, L)
        dc = self.dataconsis(L+S, d, F, mask)

        if not self.is_last:
            gamma = tf.cast(tf.nn.relu(self.gamma), tf.complex64)
        else:
            gamma = tf.cast(1.0, tf.complex64)
        M = L + S - gamma * dc
        #tf.print(gamma.numpy())

        data[0] = L
        data[1] = S
        data[2] = M

        return data
    
    def lowrank(self, L):
        Lpre=L
        St, Ut, Vt = tf.linalg.svd(L)
        #tf.print(tf.sigmoid(self.thres_coef))
        thres = tf.sigmoid(self.thres_coef) * St[:,0]
        thres = tf.expand_dims(thres, -1)
        St = tf.nn.relu(St - thres)
        St = tf.linalg.diag(St)
        
        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0,2,1])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        L = tf.linalg.matmul(US, Vt_conj) 

        return L

    def sparse(self, Mpre, L):
        M_L = tf.stack([tf.reshape(L, [self.nb, self.nt, self.nx, self.ny]), tf.reshape(Mpre, [self.nb, self.nt, self.nx, self.ny])], axis=-1)
        S = self.sconv(M_L)
        S = tf.reshape(S, [self.nb, self.nt, self.nx*self.ny])

        return S

    def dataconsis(self, LS, d, F, mask):
        resk = mask * F(tf.reshape(LS, [self.nb, self.nt, self.nx, self.ny])) - d
        dc = tf.reshape(F.TH(resk), [self.nb, self.nt, self.nx*self.ny])
        return dc
