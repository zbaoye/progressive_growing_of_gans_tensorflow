"""Some utils for MNIST dataset"""
# pylint: disable=C0301,C0103

import png
import numpy as np
import utils
import scipy.io as scio


def display_transform(image):
    image = np.squeeze(image)
    return image


def view_image(image, hparams, mask=None):
    """Process and show the image"""
    image = display_transform(image)
    if len(image) == hparams.n_input:
        image = image.reshape([28, 28])
        if mask is not None:
            mask = mask.reshape([28, 28])
            image = np.maximum(np.minimum(1.0, image - 1.0*(1-mask)), 0.0)
    utils.plot_image(image, 'Greys')


def save_image(image, path):
    """Save an matrix as a mat file"""
    
    scio.savemat(path, {'H_hat':image})
    
    H_matrix=  image[:,:,0]*image[:,:,0]+image[:,:,1]*image[:,:,1];
    scipy.misc.imsave(path, H_matrix[:,:])
