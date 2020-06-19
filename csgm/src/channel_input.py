"""Inputs for MNIST dataset"""

import math
import numpy as np
import mnist_model_def
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

NUM_TEST_IMAGES = 100


def model_input(hparams):
    """Create input tensors"""
    print(hparams)
    raw_dataset = tf.data.TFRecordDataset(hparams.dataset_dir)
    def parse_function(example_proto):
        dics = {'H_data': tf.FixedLenFeature(shape=(8192,1), dtype=tf.float32),
                'Tx_data': tf.FixedLenFeature(shape=(8192,1), dtype=tf.float32),
                'Rx_data': tf.FixedLenFeature(shape=(8192,1), dtype=tf.float32)}
        parsed_example = tf.parse_single_example(example_proto, dics)
        return parsed_example

    new_dataset = raw_dataset.map(parse_function)
    iterator = new_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    input_data ={}
    sess = tf.InteractiveSession()
    for i in range(NUM_TEST_IMAGES):
        try:
            dict_data = sess.run(next_element)
            dict_data['H_data'] = dict_data['H_data'].reshape([64,64,2],order='F')
            dict_data['Tx_data'] = dict_data['Tx_data'].reshape([64,64,2],order='F')
            dict_data['Rx_data'] = dict_data['Rx_data'].reshape([64,64,2],order='F')
            
            input_data[i] = dict_data
            
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break

    return input_data
