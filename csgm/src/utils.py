"""Some common utils"""
# pylint: disable = C0301, C0103, C0111

import os
import pickle
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scipy.misc
import scipy.io as scio

import mnist_estimators
import celebA_estimators
import channel_estimators

from sklearn.linear_model import Lasso
from l1regls import l1regls
from cvxopt import matrix

import time
import datetime
from time import gmtime, strftime

class BestKeeper(object):
    """Class to keep the best stuff"""
    def __init__(self, hparams):
        self.losses_val_best = 1e10
        self.x_hat_val_best = np.zeros(hparams.image_shape)

    def report(self, x_hat_val, losses_val):
        if losses_val < self.losses_val_best:
            self.x_hat_val_best  = x_hat_val
            self.losses_val_best = losses_val

    def get_best(self):
        return self.x_hat_val_best

def get_tf_pilot(sample_inputs):
    '''
    a1 = sample_inputs[8:120:7,0,:]
    b1 = sample_inputs[11:120:7,12,:]
    a2 = sample_inputs[8:120:7,16,:]
    b2 = sample_inputs[11:120:7,28,:]
    a3 = sample_inputs[8:120:7,32,:]
    b3 = sample_inputs[11:120:7,44,:]
    a4 = sample_inputs[8:120:7,48,:]
    b4 = sample_inputs[11:120:7,60,:]
    sample_labels = tf.concat((a1,b1,a2,b2,a3,b3,a4,b4),axis=0)
    sample_labels= tf.reshape(sample_labels,(-1,))
    return sample_labels;
    ''' 
    a = tf.concat((sample_inputs[8,0,:],sample_inputs[16,0,:],sample_inputs[24,0,:],sample_inputs[32,0,:],sample_inputs[40,0,:],sample_inputs[48,0,:]),axis=0)
    b = tf.concat((sample_inputs[12,4,:],sample_inputs[20,0,:],sample_inputs[28,0,:],sample_inputs[36,0,:],sample_inputs[44,0,:],sample_inputs[52,0,:]),axis=0)
    c = tf.concat((sample_inputs[8,8,:],sample_inputs[16,8,:],sample_inputs[24,8,:],sample_inputs[32,8,:],sample_inputs[40,8,:],sample_inputs[48,8,:]),axis=0)
    d = tf.concat((sample_inputs[12,12,:],sample_inputs[20,12,:],sample_inputs[28,12,:],sample_inputs[36,12,:],sample_inputs[44,12,:],sample_inputs[52,12,:]),axis=0)
    e = tf.concat((sample_inputs[8,16,:],sample_inputs[16,16,:],sample_inputs[24,16,:],sample_inputs[32,16,:],sample_inputs[40,16,:],sample_inputs[48,16,:]),axis=0)
    f = tf.concat((sample_inputs[12,20,:],sample_inputs[20,20,:],sample_inputs[28,20,:],sample_inputs[36,20,:],sample_inputs[44,20,:],sample_inputs[52,20,:]),axis=0)
    g = tf.concat((sample_inputs[8,24,:],sample_inputs[16,24,:],sample_inputs[24,24,:],sample_inputs[32,24,:],sample_inputs[40,24,:],sample_inputs[48,24,:]),axis=0)
    h = tf.concat((sample_inputs[12,28,:],sample_inputs[20,28,:],sample_inputs[28,28,:],sample_inputs[36,28,:],sample_inputs[44,28,:],sample_inputs[52,28,:]),axis=0)
    i = tf.concat((sample_inputs[8,32,:],sample_inputs[16,32,:],sample_inputs[24,32,:],sample_inputs[32,32,:],sample_inputs[40,32,:],sample_inputs[48,32,:]),axis=0)
    j = tf.concat((sample_inputs[12,36,:],sample_inputs[20,36,:],sample_inputs[28,36,:],sample_inputs[36,36,:],sample_inputs[44,36,:],sample_inputs[52,36,:]),axis=0)
    k = tf.concat((sample_inputs[8,40,:],sample_inputs[16,40,:],sample_inputs[24,40,:],sample_inputs[32,40,:],sample_inputs[40,40,:],sample_inputs[48,40,:]),axis=0)
    l = tf.concat((sample_inputs[12,44,:],sample_inputs[20,44,:],sample_inputs[28,44,:],sample_inputs[36,44,:],sample_inputs[44,44,:],sample_inputs[52,44,:]),axis=0)
    m = tf.concat((sample_inputs[8,48,:],sample_inputs[16,48,:],sample_inputs[24,48,:],sample_inputs[32,48,:],sample_inputs[40,48,:],sample_inputs[48,48,:]),axis=0)
    n = tf.concat((sample_inputs[12,52,:],sample_inputs[20,52,:],sample_inputs[28,52,:],sample_inputs[36,52,:],sample_inputs[44,52,:],sample_inputs[52,52,:]),axis=0)
    o = tf.concat((sample_inputs[8,56,:],sample_inputs[16,56,:],sample_inputs[24,56,:],sample_inputs[32,56,:],sample_inputs[40,56,:],sample_inputs[48,56,:]),axis=0)
    p = tf.concat((sample_inputs[12,60,:],sample_inputs[20,60,:],sample_inputs[28,60,:],sample_inputs[36,60,:],sample_inputs[44,60,:],sample_inputs[52,60,:]),axis=0)
    
    sample_labels = tf.concat((a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p),axis=0)
    return sample_labels;   
    
    
def get_pilot(sample_inputs):
    '''
    a1 = sample_inputs[8:120:7,0,:]
    b1 = sample_inputs[11:120:7,12,:]
    a2 = sample_inputs[8:120:7,16,:]
    b2 = sample_inputs[11:120:7,28,:]
    a3 = sample_inputs[8:120:7,32,:]
    b3 = sample_inputs[11:120:7,44,:]
    a4 = sample_inputs[8:120:7,48,:]
    b4 = sample_inputs[11:120:7,60,:]
    sample_labels = np.concatenate((a1,b1,a2,b2,a3,b3,a4,b4),axis=0)
    sample_labels= np.reshape(sample_labels,(-1,))
    return sample_labels;
    '''
    a = np.concatenate((sample_inputs[8,0,:],sample_inputs[16,0,:],sample_inputs[24,0,:],sample_inputs[32,0,:],sample_inputs[40,0,:],sample_inputs[48,0,:]),axis=0)
    b = np.concatenate((sample_inputs[12,4,:],sample_inputs[20,0,:],sample_inputs[28,0,:],sample_inputs[36,0,:],sample_inputs[44,0,:],sample_inputs[52,0,:]),axis=0)
    c = np.concatenate((sample_inputs[8,8,:],sample_inputs[16,8,:],sample_inputs[24,8,:],sample_inputs[32,8,:],sample_inputs[40,8,:],sample_inputs[48,8,:]),axis=0)
    d = np.concatenate((sample_inputs[12,12,:],sample_inputs[20,12,:],sample_inputs[28,12,:],sample_inputs[36,12,:],sample_inputs[44,12,:],sample_inputs[52,12,:]),axis=0)
    e = np.concatenate((sample_inputs[8,16,:],sample_inputs[16,16,:],sample_inputs[24,16,:],sample_inputs[32,16,:],sample_inputs[40,16,:],sample_inputs[48,16,:]),axis=0)
    f = np.concatenate((sample_inputs[12,20,:],sample_inputs[20,20,:],sample_inputs[28,20,:],sample_inputs[36,20,:],sample_inputs[44,20,:],sample_inputs[52,20,:]),axis=0)
    g = np.concatenate((sample_inputs[8,24,:],sample_inputs[16,24,:],sample_inputs[24,24,:],sample_inputs[32,24,:],sample_inputs[40,24,:],sample_inputs[48,24,:]),axis=0)
    h = np.concatenate((sample_inputs[12,28,:],sample_inputs[20,28,:],sample_inputs[28,28,:],sample_inputs[36,28,:],sample_inputs[44,28,:],sample_inputs[52,28,:]),axis=0)
    i = np.concatenate((sample_inputs[8,32,:],sample_inputs[16,32,:],sample_inputs[24,32,:],sample_inputs[32,32,:],sample_inputs[40,32,:],sample_inputs[48,32,:]),axis=0)
    j = np.concatenate((sample_inputs[12,36,:],sample_inputs[20,36,:],sample_inputs[28,36,:],sample_inputs[36,36,:],sample_inputs[44,36,:],sample_inputs[52,36,:]),axis=0)
    k = np.concatenate((sample_inputs[8,40,:],sample_inputs[16,40,:],sample_inputs[24,40,:],sample_inputs[32,40,:],sample_inputs[40,40,:],sample_inputs[48,40,:]),axis=0)
    l = np.concatenate((sample_inputs[12,44,:],sample_inputs[20,44,:],sample_inputs[28,44,:],sample_inputs[36,44,:],sample_inputs[44,44,:],sample_inputs[52,44,:]),axis=0)
    m = np.concatenate((sample_inputs[8,48,:],sample_inputs[16,48,:],sample_inputs[24,48,:],sample_inputs[32,48,:],sample_inputs[40,48,:],sample_inputs[48,48,:]),axis=0)
    n = np.concatenate((sample_inputs[12,52,:],sample_inputs[20,52,:],sample_inputs[28,52,:],sample_inputs[36,52,:],sample_inputs[44,52,:],sample_inputs[52,52,:]),axis=0)
    o = np.concatenate((sample_inputs[8,56,:],sample_inputs[16,56,:],sample_inputs[24,56,:],sample_inputs[32,56,:],sample_inputs[40,56,:],sample_inputs[48,56,:]),axis=0)
    p = np.concatenate((sample_inputs[12,60,:],sample_inputs[20,60,:],sample_inputs[28,60,:],sample_inputs[36,60,:],sample_inputs[44,60,:],sample_inputs[52,60,:]),axis=0)
    
    sample_labels = np.concatenate((a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p),axis=0)
    return sample_labels;       
    
    
def multiComplex(H,Tx):
    H_real = tf.slice(H,[0,0],[-1,1])
    H_imag = tf.slice(H,[0,1],[-1,1])
    Tx_real = tf.slice(Tx,[0,0],[-1,1])
    Tx_imag = tf.slice(Tx,[0,1],[-1,1])
    Rx_real = tf.subtract(tf.multiply(H_real,Tx_real) , tf.multiply(Tx_real,Tx_imag))
    Rx_imag = tf.add(tf.multiply(H_imag,Tx_real) , tf.multiply(H_real,Tx_imag))
    Rx = tf.concat([Rx_real,Rx_imag],1)
    return Rx
    
    
def calRx(H_hat,Tx,hparams):
    H_hat = tf.squeeze(H_hat*hparams.scale_factor)
    H_hat_real = tf.slice(H_hat,[0,0,0],[-1,-1,1])
    H_hat_imag = tf.slice(H_hat,[0,0,1],[-1,-1,1])
    
    Tx_real = tf.slice(Tx,[0,0,0],[-1,-1,1])
    Tx_imag = tf.slice(Tx,[0,0,1],[-1,-1,1])              
    
    H_hat_complex =  tf.complex(H_hat_real,H_hat_imag,name='H_hat_complex')
    Tx_complex = tf.complex(Tx_real,Tx_imag,name='Tx_complex')
    
    Rx_complex = tf.multiply(H_hat_complex,Tx_complex)
    Rx = tf.concat([tf.real(Rx_complex),tf.imag(Rx_complex)],axis=2)
    
    return Rx
'''
    H_real_mean = tf.reduce_mean(H_real,1)
    H_imag_mean = tf.reduce_mean(H_imag,1)
    
    Rx_real = tf.subtract(tf.multiply(H_real_mean,Tx_real) , tf.multiply(H_imag_mean,Tx_imag))
    Rx_imag = tf.add(tf.multiply(H_imag_mean,Tx_real) , tf.multiply(H_real_mean,Tx_imag))
    Rx = tf.concat([Rx_real,Rx_imag],1)
'''  
    
    
def get_l2_loss(image1, image2):
    """Get L2 loss between the two images"""
    assert image1.shape == image2.shape
    return np.mean((image1 - image2)**2)


def get_measurement_loss(x_hat, A, y):
    """Get measurement loss of the estimated image"""
    if A is None:
        y_hat = x_hat
    else:
        y_hat = multiComplex(x_hat, A)
    assert y_hat.shape == y.shape
    sess = tf.Session()
    y_hat = sess.run(y_hat)
    return np.mean((y - y_hat) ** 2)


def save_to_pickle(data, pkl_filepath):
    """Save the data to a pickle file"""
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data


def get_estimator(hparams, model_type):
    if hparams.dataset == 'mnist':
        if model_type == 'vae':
            estimator = mnist_estimators.vae_estimator(hparams)
        elif model_type == 'lasso':
            estimator = mnist_estimators.lasso_estimator(hparams)
        elif model_type == 'omp':
            estimator = mnist_estimators.omp_estimator(hparams)
        elif model_type == 'learned':
            estimator = mnist_estimators.learned_estimator(hparams)
        else:
            raise NotImplementedError
    elif hparams.dataset == 'celebA':
        if model_type == 'lasso-dct':
            estimator = celebA_estimators.lasso_dct_estimator(hparams)
        elif model_type == 'lasso-wavelet':
            estimator = celebA_estimators.lasso_wavelet_estimator(hparams)
        elif model_type == 'lasso-wavelet-ycbcr':
            estimator = celebA_estimators.lasso_wavelet_ycbcr_estimator(hparams)
        elif model_type == 'k-sparse-wavelet':
            estimator = celebA_estimators.k_sparse_wavelet_estimator(hparams)
        elif model_type == 'dcgan':
            estimator = celebA_estimators.dcgan_estimator(hparams)
        else:
            raise NotImplementedError
    elif hparams.dataset == 'channel':
        if model_type== 'vae':
            estimator = channel_estimators.vae_estimator(hparams)
        elif model_type == 'dcgan':
            estimator = channel_estimators.dcgan_estimator(hparams)
        elif model_type == 'pggan':
            estimator = channel_estimators.pggan_estimator(hparams)
        else:
            raise NotImplementedError
        
    return estimator


def get_estimators(hparams):
    estimators = {model_type: get_estimator(hparams, model_type) for model_type in hparams.model_types}
    return estimators


def setup_checkpointing(hparams):
    # Set up checkpoint directories
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        set_up_dir(checkpoint_dir)
    return checkpoint_dir

def save_channel_mat(key,h_hat,hparams):
    save_paths = hparams.checkpoint_dir + '{:08d}.mat'.format(key)
    scio.savemat(save_paths, {'H_hat':h_hat[0,:,:,:]})
    
def save_channel_image(key,h_hat,hparams):
    save_paths = hparams.checkpoint_dir + '{:08d}.png'.format(key)
    H_matrix=  h_hat[:,:,:,0]*h_hat[:,:,:,0]+h_hat[:,:,:,1]*h_hat[:,:,:,1];
    scipy.misc.imsave(save_paths, H_matrix[0,:,:])
    

def save_images(key, est_image, save_image, hparams):
    """Save a batch of images to png files"""

    save_path = get_save_paths(hparams, key)
    image = est_image.reshape(hparams.image_shape,order='F')
    save_image(image, save_path)


def checkpoint(key, est_image, measurement_losses, l2_losses, save_image, hparams):
    """Save images, measurement losses and L2 losses for a batch"""
    if hparams.save_images:
        save_images(key, est_image, save_image, hparams)

    if hparams.save_stats:
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath = get_pkl_filepaths(hparams, model_type)
            save_to_pickle(measurement_losses[model_type], m_losses_filepath)
            save_to_pickle(l2_losses[model_type], l2_losses_filepath)


def load_checkpoints(hparams):
    measurement_losses, l2_losses = {}, {}
    if hparams.save_images:
        # Load pickled loss dictionaries
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath = get_pkl_filepaths(hparams, model_type)
            measurement_losses[model_type] = load_if_pickled(m_losses_filepath)
            l2_losses[model_type] = load_if_pickled(l2_losses_filepath)
    else:
        for model_type in hparams.model_types:
            measurement_losses[model_type] = {}
            l2_losses[model_type] = {}
    return measurement_losses, l2_losses


def image_matrix(images, est_images, view_image, hparams, alg_labels=True):
    """Display images"""

    if hparams.measurement_type in ['inpaint', 'superres']:
        figure_height = 2 + len(hparams.model_types)
    else:
        figure_height = 1 + len(hparams.model_types)

    fig = plt.figure(figsize=[2*len(images), 2*figure_height])

    outer_counter = 0
    inner_counter = 0

    # Show original images
    outer_counter += 1
    for image in images.values():
        inner_counter += 1
        ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        if alg_labels:
            ax.set_ylabel('Original', fontsize=14)
        _ = fig.add_subplot(figure_height, len(images), inner_counter)
        view_image(image, hparams)

    # Show original images with inpainting mask
    if hparams.measurement_type == 'inpaint':
        mask = get_inpaint_mask(hparams)
        outer_counter += 1
        for image in images.values():
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel('Masked', fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image, hparams, mask)

    # Show original images with blurring
    if hparams.measurement_type == 'superres':
        factor = hparams.superres_factor
        A = get_A_superres(hparams)
        outer_counter += 1
        for image in images.values():
            image_low_res = np.matmul(image, A) / np.sqrt(hparams.n_input/(factor**2)) / (factor**2)
            low_res_shape = (int(hparams.image_shape[0]/factor), int(hparams.image_shape[1]/factor), hparams.image_shape[2])
            image_low_res = np.reshape(image_low_res, low_res_shape)
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel('Blurred', fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image_low_res, hparams)

    for model_type in hparams.model_types:
        outer_counter += 1
        for image in est_images[model_type].values():
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel(model_type, fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image, hparams)

    if hparams.image_matrix >= 2:
        save_path = get_matrix_save_path(hparams)
        plt.savefig(save_path)

    if hparams.image_matrix in [1, 3]:
        plt.show()


def plot_image(image, cmap=None):
    """Show the image"""
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(image, cmap=cmap)


def get_checkpoint_dir(hparams, model_type):
    base_dir = './estimated/{0}/{1}/'.format(
        hparams.dataset,
        model_type
    )

    if model_type in ['lasso', 'lasso-dct', 'lasso-wavelet','lasso-wavelet-ycbcr']:
        dir_name = '{}'.format(
            hparams.lmbd,
        )
    elif model_type == 'k-sparse-wavelet':
        dir_name = '{}'.format(
            hparams.sparsity,
        )
    elif model_type in ['vae']:
        dir_name = '{}_{}_{}'.format(
            timestamp(),
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type in ['dcgan']:
        dir_name = '{}_{}_{}'.format(
            timestamp(),
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type in ['pggan']:
        dir_name = '{}_{}_{}'.format(
            timestamp(),
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type == 'learned':
        dir_name = '50-200'
    else:
        raise NotImplementedError

    ckpt_dir = base_dir + dir_name + '/'

    return ckpt_dir


def get_pkl_filepaths(hparams, model_type):
    """Return paths for the pickle files"""
    checkpoint_dir = get_checkpoint_dir(hparams, model_type)
    m_losses_filepath = checkpoint_dir + 'measurement_losses.pkl'
    l2_losses_filepath = checkpoint_dir + 'l2_losses.pkl'
    return m_losses_filepath, l2_losses_filepath


def get_save_paths(hparams, image_num):
    checkpoint_dir = get_checkpoint_dir(hparams, 'dcgan')
    save_paths = checkpoint_dir + '{0}.mat'.format(image_num)
    return save_paths

def timestamp(s='%m%d.%H%M', ts=None):
  if not ts: ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime(s)
  return st

def get_matrix_save_path(hparams):
    save_path = './estimated/{0}/{1}/{2}/{3}/{4}/matrix_{5}.png'.format(
        hparams.dataset,
        hparams.input_type,
        hparams.measurement_type,
        hparams.noise_std,
        hparams.num_measurements,
        '_'.join(hparams.model_types)
    )
    return save_path


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)


def print_hparams(hparams):
    print ''
    for temp in dir(hparams):
        if temp[:1] != '_':
            print '{0} = {1}'.format(temp, getattr(hparams, temp))
    print ''


def get_learning_rate(global_step, hparams):
    if hparams.decay_lr:
        return tf.train.exponential_decay(hparams.learning_rate,
                                          global_step,
                                          50,
                                          0.7,
                                          staircase=True)
    else:
        return tf.constant(hparams.learning_rate)


def get_optimizer(learning_rate, hparams):
    if hparams.optimizer_type == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    if hparams.optimizer_type == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, hparams.momentum)
    elif hparams.optimizer_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate)
    elif hparams.optimizer_type == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif hparams.optimizer_type == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate)
    else:
        raise Exception('Optimizer ' + hparams.optimizer_type + ' not supported')


def get_inpaint_mask(hparams):
    image_size = hparams.image_shape[0]
    margin = (image_size - hparams.inpaint_size) / 2
    mask = np.ones(hparams.image_shape)
    mask[margin:margin+hparams.inpaint_size, margin:margin+hparams.inpaint_size] = 0
    return mask


def get_A_inpaint(hparams):
    mask = get_inpaint_mask(hparams)
    mask = mask.reshape(1, -1)
    A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
    A = np.asarray([a for a in A if np.sum(a) != 0])

    # Make sure that the norm of each row of A is hparams.n_input
    A = np.sqrt(hparams.n_input) * A
    assert all(np.abs(np.sum(A**2, 1) - hparams.n_input) < 1e-6)

    return A.T


def get_A_superres(hparams):
    factor = hparams.superres_factor
    A = np.zeros((int(hparams.n_input/(factor**2)), hparams.n_input))
    l = 0
    for i in range(hparams.image_shape[0]/factor):
        for j in range(hparams.image_shape[1]/factor):
            for k in range(hparams.image_shape[2]):
                a = np.zeros(hparams.image_shape)
                a[factor*i:factor*(i+1), factor*j:factor*(j+1), k] = 1
                A[l, :] = np.reshape(a, [1, -1])
                l += 1

    # Make sure that the norm of each row of A is hparams.n_input
    A = np.sqrt(hparams.n_input/(factor**2)) * A
    assert all(np.abs(np.sum(A**2, 1) - hparams.n_input) < 1e-6)

    return A.T


def get_A_restore_path(hparams):
    pattern = './optimization/mnist-e2e/checkpoints/adam_0.001_{0}_{1}/'
    if hparams.measurement_type == 'fixed':
        ckpt_dir = pattern.format(hparams.num_measurements, 'False')
    elif hparams.measurement_type == 'learned':
        ckpt_dir = pattern.format(hparams.num_measurements, 'True')
    else:
        raise NotImplementedError
    restore_path = tf.train.latest_checkpoint(ckpt_dir)
    return restore_path


def restore_A(hparams):
    A = tf.get_variable('A', [784, hparams.num_measurements])
    restore_path = get_A_restore_path(hparams)
    model_saver = tf.train.Saver([A])
    with tf.Session() as sess:
        model_saver.restore(sess, restore_path)
        A_val = sess.run(A)
    tf.reset_default_graph()
    return A_val


def get_A(hparams):
    if hparams.measurement_type == 'gaussian':
        A = np.random.randn(hparams.n_input, hparams.num_measurements)
    elif hparams.measurement_type == 'superres':
        A = get_A_superres(hparams)
    elif hparams.measurement_type in ['fixed', 'learned']:
        A = restore_A(hparams)
    elif hparams.measurement_type == 'inpaint':
        A = get_A_inpaint(hparams)
    elif hparams.measurement_type == 'project':
        A = None
    else:
        raise NotImplementedError
    return A


def set_num_measurements(hparams):
    if hparams.measurement_type == 'project':
        hparams.num_measurements = hparams.n_input
    elif hparams.measurement_type == 'pilot':
        hparams.num_measurements = hparams.num_measurements
    else:
        hparams.num_measurements = get_A(hparams).shape[1]


def get_checkpoint_path(ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = os.path.join(ckpt_dir,
                                 ckpt.model_checkpoint_path)
    else:
        print 'No checkpoint file found'
        ckpt_path = ''
    return ckpt_path


def RGB_matrix():
    U_ = np.zeros((12288, 12288))
    U = np.zeros((3, 3))
    V = np.zeros((12288, 1))

    # R, Y
    V[0::3] = ((255.0/219.0)*(-16.0)) + ((255.0*0.701/112.0)*(-128.0))
    U[0, 0] = (255.0/219.0)
    # R, Cb
    U[0, 1] = (0.0)
    # R, Cr
    U[0, 2] = (255.0*0.701/112.0)

    # G, Y
    V[1::3] = ((255.0/219.0)*(-16.0)) - ((0.886*0.114*255.0/(112.0*0.587)) *(-128.0)) - ((255.0*0.701*0.299/(112.0*0.587))*(-128.0))
    U[1, 0] = (255.0/219.0)
    # G, Cb
    U[1, 1] = - (0.886*0.114*255.0/(112.0*0.587))  #*np.eye(4096)
    # G, Cr
    U[1, 2] = - (255.0*0.701*0.299/(112.0*0.587))  #*np.eye(4096)

    # B, Y
    V[2::3] = ((255.0/219.0)*(-16.0)) + ((0.886*255.0/(112.0))*(-128.0))
    U[2, 0] = (255.0/219.0)  #*np.eye(4096)
    # B, Cb
    U[2, 1] = (0.886*255.0/(112.0))  #*np.eye(4096)
    # B, Cr
    U[2, 2] = 0.0

    for i in range(4096):
        U_[i*3:(i+1)*3, i*3:(i+1)*3] = U
    return U_, V


def YCbCr(image):
    """
     input: array with RGB values between 0 and 255
     output: array with YCbCr values between 16 and 235(Y) or 240(Cb, Cr)
    """
    x = image.copy()
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    # Y channel = 16.0 + 65.378/256 R + 129.057/256 * G + 25.064/256.0 * B
    x[:, :, 0] = 16.0 + (65.738/256.0)*R + (129.057/256.0)*G + (25.064/256.0)*B
    # Cb channel = 128.0 - 37.945/256 R - 74.494/256 * G + 112.439/256 * B
    x[:, :, 1] = 128.0 - (37.945/256.0)*R - (74.494/256.0)*G + (112.439/256.0)*B
    # Cr channel = 128.0+ 112.439/256 R - 94.154/256 * G - 18.285/256 * B
    x[:, :, 2] = 128.0 + (112.439/256.0)*R - (94.154/256.0)*G - (18.285/256.0)*B
    return x


def RGB(image):
    """
     input: array with YCbCr values between 16 and 235(Y) or 240(Cb, Cr)
     output: array with RGB values between 0 and 255
    """
    x = image.copy()
    Y = image[:, :, 0]
    Cb = image[:, :, 1]
    Cr = image[:, :, 2]
    x[:, :, 0] = (255.0/219.0)*(Y - 16.0) + (0.0/112.0) *(Cb - 128.0)+ (255.0*0.701/112.0)*(Cr - 128.0)
    x[:, :, 1] = (255.0/219.0)*(Y - 16.0) - (0.886*0.114*255.0/(112.0*0.587)) *(Cb - 128.0) - (255.0*0.701*0.299/(112.0*0.587))*(Cr - 128.0)
    x[:, :, 2] = (255.0/219.0)*(Y - 16.0) + (0.886*255.0/(112.0)) *(Cb - 128.0) + (0.0/112.0)*(Cr - 128.0)
    return x


def save_plot(is_save, save_path):
    if is_save:
        pdf = PdfPages(save_path)
        pdf.savefig(bbox_inches='tight')
        pdf.close()


def solve_lasso(A_val, y_val, hparams):
    if hparams.lasso_solver == 'sklearn':
        lasso_est = Lasso(alpha=hparams.lmbd)
        lasso_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
        x_hat = lasso_est.coef_
        x_hat = np.reshape(x_hat, [-1])
    if hparams.lasso_solver == 'cvxopt':
        A_mat = matrix(A_val.T)
        y_mat = matrix(y_val)
        x_hat_mat = l1regls(A_mat, y_mat)
        x_hat = np.asarray(x_hat_mat)
        x_hat = np.reshape(x_hat, [-1])
    return x_hat


def get_opt_reinit_op(opt, var_list, global_step):
    opt_slots = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in var_list]
    #if isinstance(opt, tf.train.AdamOptimizer):
    #    opt_slots.extend([opt._beta1_power, opt._beta2_power])  #pylint: disable = W0212
    all_opt_variables = opt_slots + var_list + [global_step]
    opt_reinit_op = tf.variables_initializer(all_opt_variables)
    return opt_reinit_op
