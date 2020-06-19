"""Compressed sensing main script"""
# pylint: disable=C0301,C0103,C0111

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(hparams):

    # Set up some stuff accoring to hparams
    hparams.n_input = np.prod(hparams.image_shape)
    utils.set_num_measurements(hparams)
    utils.print_hparams(hparams)

    # get inputs
    data_dict = model_input(hparams)

    estimator = utils.get_estimator(hparams,hparams.model_types[0])
    print(estimator)
    hparams.checkpoint_dir = utils.setup_checkpointing(hparams)
    measurement_losses, l2_losses = utils.load_checkpoints(hparams)

    h_hats_dict = {model_type : {} for model_type in hparams.model_types}
    for key, x in data_dict.iteritems():
        if not hparams.not_lazy:
            # If lazy, first check if the image has already been
            # saved before by *all* estimators. If yes, then skip this image.
            save_paths = utils.get_save_paths(hparams, key)
            is_saved = all([os.path.isfile(save_path) for save_path in save_paths.values()])
            if is_saved:
                continue

        # Get Rx data
        Rx = data_dict[key]['Rx_data']
        Tx = data_dict[key]['Tx_data']
        H = data_dict[key]['H_data']
        Pilot_Rx = utils.get_pilot(Rx)
        print('Pilot_shape',Pilot_Rx.shape)
        Pilot_Rx = Pilot_Rx[0::2]+Pilot_Rx[1::2]*1j
        Pilot_Tx = utils.get_pilot(Tx)
        Pilot_Tx = Pilot_Tx[0::2]+Pilot_Tx[1::2]*1j
        Pilot_complex = Pilot_Rx/Pilot_Tx
        Pilot = np.empty((Pilot_complex.size*2,), dtype=Pilot_Rx.dtype)
        Pilot[0::2] = np.real(Pilot_complex)
        Pilot[1::2] = np.imag(Pilot_complex)
        
        
        Pilot = np.reshape(Pilot,[1,-1])/2.5
        # Construct estimates using each estimator
        h_hat = estimator(Tx, Rx, Pilot, hparams)


        # Compute and store measurement and l2 loss
#        measurement_losses['dcgan'][key] = utils.get_measurement_loss(h_hat, Tx, Rx)
#        l2_losses['dcgan'][key] = utils.get_l2_loss(h_hat, H)

        
        print "Processed upto image {0} / {1}".format(key+1, len(data_dict))

        # Checkpointing
        if (hparams.save_images) and ((key+1) % hparams.checkpoint_iter == 0):
            # utils.checkpoint(key,h_hat, measurement_losses, l2_losses, save_image, hparams)
            utils.save_channel_image(key+1,h_hat,hparams)
            utils.save_channel_mat(key+1,h_hat,hparams)
            
            
            print '\nProcessed and saved first ', key+1, 'channels\n'



if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./models/celebA_64_64/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--dataset-dir', type=str, default='', help='Dataset Dir')
    PARSER.add_argument('--input-height', type=int, default='64', help='Data Input Height')
    PARSER.add_argument('--input-width', type=int, default='64', help='Data Input Width')
    PARSER.add_argument('--input-channel', type=int, default='2', help='Data Input Channel')
    PARSER.add_argument('--input-type', type=str, default='random_test', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='./data/celebAtest/*.jpg', help='Pattern to match to get images')
    PARSER.add_argument('--num-input-images', type=int, default=10, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=64, help='How many examples are processed together')
    PARSER.add_argument('--scale-factor', type=float, default=2.5, help='set H data scale to orign According to DCGAN Setting')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--inpaint-size', type=int, default=1, help='size of block to inpaint')
    PARSER.add_argument('--superres-factor', type=int, default=2, help='how downsampled is the image')

    # Model
    PARSER.add_argument('--z-dim', type=int, default=100, help='z dim')
    PARSER.add_argument('--pilot-dim', type=int, default=100, help='pilot dim')
    PARSER.add_argument('--model-types', type=str, nargs='+', default=None, help='model(s) used for estimation')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=0.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.0, help='weight on z prior')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='momentum', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=100, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=10, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')

    # LASSO specific hparams
    PARSER.add_argument('--lmbd', type=float, default=0.1, help='lambda : regularization parameter for LASSO')
    PARSER.add_argument('--lasso-solver', type=str, default='sklearn', help='Solver for LASSO')

    # k-sparse-wavelet specific hparams
    PARSER.add_argument('--sparsity', type=int, default=1, help='number of non zero entries allowed in k-sparse-wavelet')

    # Output
    PARSER.add_argument('--not-lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=0,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                       )
    PARSER.add_argument('--gif', action='store_true', help='whether to create a gif')
    PARSER.add_argument('--gif-iter', type=int, default=1, help='save gif frame every x iter')
    PARSER.add_argument('--gif-dir', type=str, default='', help='where to store gif frames')

    HPARAMS = PARSER.parse_args()

    if HPARAMS.dataset == 'mnist':
        HPARAMS.image_shape = (28, 28, 1)
        from mnist_input import model_input
        from mnist_utils import view_image, save_image
    elif HPARAMS.dataset == 'celebA':
        HPARAMS.image_shape = (64, 64, 3)
        from celebA_input import model_input
        from celebA_utils import view_image, save_image
    elif HPARAMS.dataset == 'channel':
        HPARAMS.image_shape = (HPARAMS.input_height, HPARAMS.input_width, HPARAMS.input_channel)
        HPARAMS.modSignal_shape = (HPARAMS.input_height, HPARAMS.input_width, HPARAMS.input_channel)
        from channel_input import model_input
        from channel_utils import view_image, save_image
    else:
        raise NotImplementedError

    main(HPARAMS)
