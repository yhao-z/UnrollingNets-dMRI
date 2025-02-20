import tensorflow as tf
import os
import scipy.io as scio
import numpy as np
import sys
import glob
from loguru import logger


# load data
def get_dataset(mode, datadir, batch_size, shuffle=False):
    filenames = glob.glob(datadir+mode+'/*.tfrecord')
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(singCoil_parse_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=50)
    dataset = dataset.batch(batch_size)

    return dataset


def parse_function(example_proto):
    dics = {'k_real': tf.io.VarLenFeature(dtype=tf.float32),
            'k_imag': tf.io.VarLenFeature(dtype=tf.float32),
            'csm_real': tf.io.VarLenFeature(dtype=tf.float32),
            'csm_imag': tf.io.VarLenFeature(dtype=tf.float32),
            'k_shape': tf.io.VarLenFeature(dtype=tf.int64),
            'csm_shape': tf.io.VarLenFeature(dtype=tf.int64)}
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    parsed_example['k_real'] = tf.sparse.to_dense(parsed_example['k_real'])
    parsed_example['k_imag'] = tf.sparse.to_dense(parsed_example['k_imag'])
    parsed_example['csm_real'] = tf.sparse.to_dense(parsed_example['csm_real'])
    parsed_example['csm_imag'] = tf.sparse.to_dense(parsed_example['csm_imag'])
    parsed_example['k_shape'] = tf.sparse.to_dense(parsed_example['k_shape'])
    parsed_example['csm_shape'] = tf.sparse.to_dense(parsed_example['csm_shape'])

    k = tf.complex(parsed_example['k_real'], parsed_example['k_imag'])
    csm = tf.complex(parsed_example['csm_real'], parsed_example['csm_imag'])

    k = tf.reshape(k, parsed_example['k_shape'])
    csm = tf.reshape(csm, parsed_example['csm_shape'])

    return k, csm


def singCoil_parse_function(example_proto):
    dics = {'k_real': tf.io.VarLenFeature(dtype=tf.float32),
            'k_imag': tf.io.VarLenFeature(dtype=tf.float32),
            'label_real': tf.io.VarLenFeature(dtype=tf.float32),
            'label_imag': tf.io.VarLenFeature(dtype=tf.float32),
            'k_shape': tf.io.VarLenFeature(dtype=tf.int64),
            'label_shape': tf.io.VarLenFeature(dtype=tf.int64)}
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    parsed_example['k_real'] = tf.sparse.to_dense(parsed_example['k_real'])
    parsed_example['k_imag'] = tf.sparse.to_dense(parsed_example['k_imag'])
    parsed_example['label_real'] = tf.sparse.to_dense(parsed_example['label_real'])
    parsed_example['label_imag'] = tf.sparse.to_dense(parsed_example['label_imag'])
    parsed_example['k_shape'] = tf.sparse.to_dense(parsed_example['k_shape'])
    parsed_example['label_shape'] = tf.sparse.to_dense(parsed_example['label_shape'])

    k = tf.complex(parsed_example['k_real'], parsed_example['k_imag'])
    label = tf.complex(parsed_example['label_real'], parsed_example['label_imag'])

    k = tf.reshape(k, parsed_example['k_shape'])
    label = tf.reshape(label, parsed_example['label_shape'])

    return k, label


# load data
def get_dataset_multicoil(mode, datadir, batch_size, shuffle=False):
    filenames = glob.glob(datadir+mode+'/*.tfrecord')
        
    logger.info(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(multicoil_parse_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=20)
    dataset = dataset.batch(batch_size)

    return dataset


def multicoil_parse_function(example_proto):
    dics = {'k_real': tf.io.VarLenFeature(dtype=tf.float32),
            'k_imag': tf.io.VarLenFeature(dtype=tf.float32),
            'csm_real': tf.io.VarLenFeature(dtype=tf.float32),
            'csm_imag': tf.io.VarLenFeature(dtype=tf.float32),
            'k_shape': tf.io.VarLenFeature(dtype=tf.int64),
            'csm_shape': tf.io.VarLenFeature(dtype=tf.int64)}
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    parsed_example['k_real'] = tf.sparse.to_dense(parsed_example['k_real'])
    parsed_example['k_imag'] = tf.sparse.to_dense(parsed_example['k_imag'])
    parsed_example['csm_real'] = tf.sparse.to_dense(parsed_example['csm_real'])
    parsed_example['csm_imag'] = tf.sparse.to_dense(parsed_example['csm_imag'])
    parsed_example['k_shape'] = tf.sparse.to_dense(parsed_example['k_shape'])
    parsed_example['csm_shape'] = tf.sparse.to_dense(parsed_example['csm_shape'])

    k = tf.complex(parsed_example['k_real'], parsed_example['k_imag'])
    csm = tf.complex(parsed_example['csm_real'], parsed_example['csm_imag'])

    k = tf.reshape(k, parsed_example['k_shape'])
    csm = tf.reshape(csm, parsed_example['csm_shape'])

    return k, csm
