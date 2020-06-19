"""Model definitions for MNIST"""
# pylint: disable = C0301, C0103, R0914, C0111

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mnist_vae.src import model_def as mnist_vae_model_def
from mnist_e2e.model_def import end_to_end

from celebA_dcgan import model_def as celebA_dcgan_model_def
import channel_pggan_def as pggan_model_def

def pggan_gen(z, pilot, hparams):

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
    z_full = tf.zeros([64, hparams.z_dim]) + z
    
    pilot_full = tf.tile(pilot,[64,1])
    #z_full = tf.concat([z_full, pilot_full],1) # conditional

    model_hparams = pggan_model_def.Hparams()

    x_hat_full = pggan_model_def.generator(model_hparams, z_full, train=False, reuse=False)
    x_hat_batch = x_hat_full[:hparams.batch_size]

    restore_vars = pggan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return x_hat_batch, restore_dict, restore_path

def dcgan_discrim(x_hat_batch, pilot, hparams):

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
    x_hat_image = tf.reshape(x_hat_batch, [-1, 64, 16, 2])
    all_zeros = tf.zeros([64, 64, 16, 2])
    discrim_input = all_zeros + x_hat_image

    yb = tf.reshape(pilot, [hparams.batch_size, 1, 1, hparams.pilot_dim]) # conditional
    discrim_input = conv_cond_concat(discrim_input, yb) # conditional
    
    model_hparams = celebA_dcgan_model_def.Hparams()
    prob, _ = celebA_dcgan_model_def.discriminator(model_hparams, discrim_input, train=False, reuse=False)
    prob = tf.reshape(prob, [-1])
    prob = prob[:hparams.batch_size]

    restore_vars = celebA_dcgan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return prob, restore_dict, restore_path

def dcgan_gen(z, pilot, hparams):

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
    z_full = tf.zeros([64, hparams.z_dim]) + z
    
    pilot_full = tf.tile(pilot,[64,1])
    z_full = tf.concat([z_full, pilot_full],1) # conditional

    model_hparams = celebA_dcgan_model_def.Hparams()

    x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=False)
    x_hat_batch = x_hat_full[:hparams.batch_size]

    restore_vars = celebA_dcgan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return x_hat_batch, restore_dict, restore_path


def construct_gen(hparams, model_def):

    model_hparams = model_def.Hparams()

    z = model_def.get_z_var(model_hparams, hparams.batch_size)
    x_hat,_  = model_def.generator(model_hparams, z, 'gen', reuse=False)

    restore_vars = model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)
    x_hat = tf.transpose(tf.reshape(x_hat,[2,-1]))

    return z, x_hat, restore_path, restore_dict


def vae_gen(hparams):
    return construct_gen(hparams, mnist_vae_model_def)

def conv_cond_concat(x, y):
    # Concatenate conditioning vector on feature map axis.
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

