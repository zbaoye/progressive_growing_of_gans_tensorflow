# pylint: disable = C0103, C0111, C0301, R0914

"""Model definitions for celebA

This file is partially based on
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/main.py
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

They come with the following license: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
"""

import tensorflow as tf
import ops


class Hparams(object):
    def __init__(self):
        self.c_dim = 2
        self.z_dim = 128
        self.output_height = 64
        self.output_width = 16
        self.gf_dim = 64
        self.df_dim = 64
        self.gfc_dim = 1024
        self.dfc_dim = 1024
        self.batch_size = 64


def generator(hparams, z, train, reuse):
    with tf.variable_scope("generator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s_h, s_w = hparams.output_height, hparams.output_width
        s_h2, s_h4, s_h8, s_h16 = int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
        s_w2, s_w4, s_w8, s_w16 = int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

        g_bn0 = ops.batch_norm(name='g_bn0')
        g_bn1 = ops.batch_norm(name='g_bn1')
        g_bn2 = ops.batch_norm(name='g_bn2')
        g_bn3 = ops.batch_norm(name='g_bn3')

        # project `z` and reshape
        h0 = tf.reshape(ops.linear(z, hparams.gf_dim*8*s_h16*s_w16, 'g_h0_lin'), [-1, s_h16, s_w16, hparams.gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0, train=train))

        h1 = ops.deconv2d(h0, [hparams.batch_size, s_h8, s_w8, hparams.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(g_bn1(h1, train=train))

        h2 = ops.deconv2d(h1, [hparams.batch_size, s_h4, s_w4, hparams.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2, train=train))

        h3 = ops.deconv2d(h2, [hparams.batch_size, s_h2, s_w2, hparams.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3, train=train))

        h4 = ops.deconv2d(h3, [hparams.batch_size, s_h, s_w, hparams.c_dim], name='g_h4')
        x_gen = tf.nn.tanh(h4)

        return x_gen


def discriminator(hparams, x, train, reuse):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        d_bn1 = ops.batch_norm(name='d_bn1')
        d_bn2 = ops.batch_norm(name='d_bn2')
        d_bn3 = ops.batch_norm(name='d_bn3')

        h0 = ops.lrelu(ops.conv2d(x, hparams.df_dim, name='d_h0_conv'))

        h1 = ops.conv2d(h0, hparams.df_dim*2, name='d_h1_conv')
        h1 = ops.lrelu(d_bn1(h1, train=train))

        h2 = ops.conv2d(h1, hparams.df_dim*4, name='d_h2_conv')
        h2 = ops.lrelu(d_bn2(h2, train=train))

        h3 = ops.conv2d(h2, hparams.df_dim*8, name='d_h3_conv')
        h3 = ops.lrelu(d_bn3(h3, train=train))

        h4 = ops.linear(tf.reshape(h3, [hparams.batch_size, -1]), 1, 'd_h3_lin')

        d_logit = h4
        d = tf.nn.sigmoid(d_logit)

        return d, d_logit


def gen_restore_vars():
    restore_vars = ['generator/g_bn0/beta',
                    'generator/g_bn0/gamma',
                    'generator/g_bn0/moving_mean',
                    'generator/g_bn0/moving_variance',
                    'generator/g_bn1/beta',
                    'generator/g_bn1/gamma',
                    'generator/g_bn1/moving_mean',
                    'generator/g_bn1/moving_variance',
                    'generator/g_bn2/beta',
                    'generator/g_bn2/gamma',
                    'generator/g_bn2/moving_mean',
                    'generator/g_bn2/moving_variance',
                    'generator/g_bn3/beta',
                    'generator/g_bn3/gamma',
                    'generator/g_bn3/moving_mean',
                    'generator/g_bn3/moving_variance',
                    'generator/g_h0_lin/Matrix',
                    'generator/g_h0_lin/bias',
                    'generator/g_h1/biases',
                    'generator/g_h1/w',
                    'generator/g_h2/biases',
                    'generator/g_h2/w',
                    'generator/g_h3/biases',
                    'generator/g_h3/w',
                    'generator/g_h4/biases',
                    'generator/g_h4/w']
    return restore_vars



def discrim_restore_vars():
    restore_vars = ['discriminator/d_bn1/beta',
                    'discriminator/d_bn1/gamma',
                    'discriminator/d_bn1/moving_mean',
                    'discriminator/d_bn1/moving_variance',
                    'discriminator/d_bn2/beta',
                    'discriminator/d_bn2/gamma',
                    'discriminator/d_bn2/moving_mean',
                    'discriminator/d_bn2/moving_variance',
                    'discriminator/d_bn3/beta',
                    'discriminator/d_bn3/gamma',
                    'discriminator/d_bn3/moving_mean',
                    'discriminator/d_bn3/moving_variance',
                    'discriminator/d_h0_conv/biases',
                    'discriminator/d_h0_conv/w',
                    'discriminator/d_h1_conv/biases',
                    'discriminator/d_h1_conv/w',
                    'discriminator/d_h2_conv/biases',
                    'discriminator/d_h2_conv/w',
                    'discriminator/d_h3_conv/biases',
                    'discriminator/d_h3_conv/w',
                    'discriminator/d_h3_lin/Matrix',
                    'discriminator/d_h3_lin/bias']
    return restore_vars
