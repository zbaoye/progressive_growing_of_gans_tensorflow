

"""Model definitions for PG-GAN


"""

import tensorflow as tf
import numpy as np
from ops import lrelu, conv2d, fully_connect, upscale, Pixl_Norm, downscale2d, MinibatchstateConcat


class Hparams(object):
    def __init__(self):

        self.batch_size = 64
        self.use_wscale = True


def generator(hparams, z_var, train, reuse):
    with tf.variable_scope("generator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        use_wscale = True
        de = tf.reshape(Pixl_Norm(z_var), [hparams.batch_size, 1, 1, 128])
        de = conv2d(de, output_dim=128, k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
        de = Pixl_Norm(lrelu(de))
        de = tf.reshape(de, [hparams.batch_size, 4, 4, 128])
        de = conv2d(de, output_dim=128, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_2_conv')
        de = Pixl_Norm(lrelu(de))

        de = upscale(de, 2)
        de = Pixl_Norm(lrelu(conv2d(de, output_dim=128, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_conv_1_8')))
        de = Pixl_Norm(lrelu(conv2d(de, output_dim=128, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_conv_2_8')))
        
        de = upscale(de, 2)
        de = Pixl_Norm(lrelu(conv2d(de, output_dim=64, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_conv_1_16')))
        de = Pixl_Norm(lrelu(conv2d(de, output_dim=64, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_conv_2_16')))
        
        de = upscale(de, 2)
        de = Pixl_Norm(lrelu(conv2d(de, output_dim=32, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_conv_1_32')))
        de = Pixl_Norm(lrelu(conv2d(de, output_dim=32, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_conv_2_32')))
            
        de_iden = conv2d(de, output_dim=2, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=use_wscale,name='gen_y_rgb_conv_32')
        de_iden = upscale(de_iden, 2)
            
            
        de = upscale(de, 2)
        de = Pixl_Norm(lrelu(conv2d(de, output_dim=16, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_conv_1_64')))
        de = Pixl_Norm(lrelu(conv2d(de, output_dim=16, d_w=1, d_h=1, use_wscale=use_wscale, name='gen_n_conv_2_64')))
        
        de = conv2d(de, output_dim=2, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=use_wscale, gain=1, name='gen_y_rgb_conv_64')
        
        return de



def gen_restore_vars():
    restore_vars = ['generator/gen_n_1_conv/weight',
                    'generator/gen_n_1_conv/biases',
                    'generator/gen_n_2_conv/weight',
                    'generator/gen_n_2_conv/biases',
                    'generator/gen_n_conv_1_8/weight',
                    'generator/gen_n_conv_1_8/biases',
                    'generator/gen_n_conv_2_8/weight',
                    'generator/gen_n_conv_2_8/biases',
                    'generator/gen_n_conv_1_16/weight',
                    'generator/gen_n_conv_1_16/biases',
                    'generator/gen_n_conv_2_16/weight',
                    'generator/gen_n_conv_2_16/biases',
                    'generator/gen_n_conv_1_32/weight',
                    'generator/gen_n_conv_1_32/biases',
                    'generator/gen_n_conv_2_32/weight',
                    'generator/gen_n_conv_2_32/biases',
                    'generator/gen_y_rgb_conv_32/weight',
                    'generator/gen_y_rgb_conv_32/biases',
                    'generator/gen_n_conv_1_64/weight',
                    'generator/gen_n_conv_1_64/biases',
                    'generator/gen_n_conv_2_64/weight',
                    'generator/gen_n_conv_2_64/biases',
                    'generator/gen_y_rgb_conv_64/weight',
                    'generator/gen_y_rgb_conv_64/biases']
    return restore_vars




