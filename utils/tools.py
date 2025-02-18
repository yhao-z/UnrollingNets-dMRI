import tempfile
import os
import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.io as scio


def loss_SLR_ISTA(y, y_, y_sym):
    pred = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
    label = tf.stack([tf.math.real(y_), tf.math.imag(y_)], axis=-1)

    cost = tf.reduce_mean(tf.math.square(pred - label))
    cost_sym = 0
    for k in range(len(y_sym)):
        cost_sym += tf.reduce_mean(tf.square(y_sym[k]))

    loss = cost + 0.01 * cost_sym
    return loss


def calc_SNR(recon, label):
    recon = np.array(recon).flatten()
    label = np.array(label).flatten()
    err = np.linalg.norm(label - recon) ** 2
    snr = 10 * np.log10(np.linalg.norm(label) ** 2 / err)

    return snr


def calc_PSNR(recon, label):
    recon = np.array(recon).flatten()
    label = np.array(label).flatten()
    err = np.linalg.norm(label - recon) ** 2
    max_label = np.max(np.abs(label))
    N = np.prod(recon.shape)
    psnr = 10 * np.log10(N * max_label ** 2 / err)

    return psnr


def mse(recon, label):
    if recon.dtype == tf.complex64:
        residual_cplx = recon - label
        residual = tf.stack([tf.math.real(residual_cplx), tf.math.imag(residual_cplx)], axis=-1)
        mse = tf.reduce_mean(residual ** 2)
    else:
        residual = recon - label
        mse = tf.reduce_mean(residual ** 2)
    return mse


def fft2c_mri(xin):
    # nx, ny are the last two dimensions
    out = tf.signal.ifftshift(xin, [-2,-1])
    out = tf.signal.fft2d(out)
    out = tf.signal.fftshift(out, [-2,-1])
    [nx, ny] = xin.shape[-2:]
    out = out / tf.sqrt(tf.cast(nx*ny, tf.complex64))

    return out


def ifft2c_mri(xin):
    # nx, ny are the last two dimensions
    out = tf.signal.ifftshift(xin, [-2,-1])
    out = tf.signal.ifft2d(out)
    out = tf.signal.fftshift(out, [-2,-1])
    [nx, ny] = xin.shape[-2:]
    out = out * tf.sqrt(tf.cast(nx*ny, tf.complex64))

    return out


def rsos(x):
    # x: nb, ncoil, nt, nx, ny; complex64
    x = tf.sqrt(tf.math.reduce_sum(tf.abs(x ** 2), axis=1))
    return x


class mriF(object):
    def __init__(self, csm=None):
        self.csm = csm

    def __call__(self, x):
        if self.csm == None:
            X = fft2c_mri(x)
        else: 
            x = tf.expand_dims(x, 1)
            x = self.csm * x
            X = fft2c_mri(x)
        return X

    def TH(self, X):
        if self.csm == None:
            x = ifft2c_mri(X)
        else:
            x = ifft2c_mri(X)
            x = x * tf.math.conj(self.csm)
            x = tf.reduce_sum(x, 1)
        return x
