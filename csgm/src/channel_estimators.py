"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
import tensorflow as tf
import channel_model_def
from mnist_utils import save_image
import utils


def lasso_estimator(hparams):  # pylint: disable = W0613
    """LASSO estimator"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            x_hat = utils.solve_lasso(A_val, y_val, hparams)
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def omp_estimator(hparams):
    """OMP estimator"""
    omp_est = OrthogonalMatchingPursuit(n_nonzero_coefs=hparams.omp_k)
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            omp_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
            x_hat = omp_est.coef_
            x_hat = np.reshape(x_hat, [-1])
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator

def pggan_estimator(hparams):
    # pylint: disable = C0326

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    Tx = tf.placeholder(tf.float32, shape=hparams.modSignal_shape, name='Tx')
    Rx = tf.placeholder(tf.float32, shape=hparams.modSignal_shape, name='Rx')
    Pilot = tf.placeholder(tf.float32, shape=[hparams.batch_size, hparams.pilot_dim], name='Pilot')

    # Create the generator
    z_batch = tf.Variable(tf.random.normal([hparams.batch_size, hparams.z_dim]), name='z_batch')
    H_hat, restore_dict_gen, restore_path_gen = channel_model_def.pggan_gen(z_batch, Pilot, hparams)

    # measure the estimate
    print('H_hat:',H_hat.shape)
    print('Tx:',Tx.shape)
    Rx_hat = utils.calRx(H_hat,Tx,hparams)
    '''
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y2_batch')
    elif hparams.measurement_type == 'pilot':
        Rx_hat = utils.calRx(H_hat,Tx,hparams)
        # Rx_hat = utils.multiComplex(H_hat,Tx);
        # Rx_hat = tf.multiply(H_hat, Tx, name='y_hat')  # TODO complex mult
    else:
        measurement_is_sparse = (hparams.measurement_type in ['inpaint', 'superres'])
        y_hat_batch = tf.matmul(x_hat_batch, A, b_is_sparse=measurement_is_sparse, name='y2_batch')
    '''
    
    # define all losses
    if hparams.measurement_type == 'pilot':
        # only polit Loss
        m_loss1_batch =  tf.abs(utils.get_tf_pilot(Rx) - utils.get_tf_pilot(Rx_hat))
        m_loss2_batch =  (utils.get_tf_pilot(Rx) - utils.get_tf_pilot(Rx_hat))**2
        zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
    else:
        m_loss1_batch =  tf.reduce_mean(tf.abs(Rx - Rx_hat), 1)
        m_loss2_batch =  tf.reduce_mean((Rx - Rx_hat)**2, 1)
        zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
        
    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_gen.restore(sess, restore_path_gen)

    def estimator(Tx_val, Rx_val, Pilot_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)

        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {Tx: Tx_val, Rx: Rx_val, Pilot: Pilot_val}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                print logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val)

            H_hat_val, total_loss_val = sess.run([H_hat, total_loss], feed_dict=feed_dict)
            best_keeper.report(H_hat_val, total_loss_val)
        return best_keeper.get_best()

    return estimator

def dcgan_estimator(hparams):
    # pylint: disable = C0326

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    Tx = tf.placeholder(tf.float32, shape=hparams.modSignal_shape, name='Tx')
    Rx = tf.placeholder(tf.float32, shape=hparams.modSignal_shape, name='Rx')
    Pilot = tf.placeholder(tf.float32, shape=[hparams.batch_size, hparams.pilot_dim], name='Pilot')

    # Create the generator
    z_batch = tf.Variable(tf.random.uniform([hparams.batch_size, hparams.z_dim]), name='z_batch')
    H_hat, restore_dict_gen, restore_path_gen = channel_model_def.dcgan_gen(z_batch, Pilot, hparams)

    # Create the discriminator
    #prob, restore_dict_discrim, restore_path_discrim = channel_model_def.dcgan_discrim(H_hat, Pilot, hparams)

    # measure the estimate
    Rx_hat = utils.calRx(H_hat,Tx,hparams)
    '''
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y2_batch')
    elif hparams.measurement_type == 'pilot':
        Rx_hat = utils.calRx(H_hat,Tx,hparams)
        # Rx_hat = utils.multiComplex(H_hat,Tx);
        # Rx_hat = tf.multiply(H_hat, Tx, name='y_hat')  # TODO complex mult
    else:
        measurement_is_sparse = (hparams.measurement_type in ['inpaint', 'superres'])
        y_hat_batch = tf.matmul(x_hat_batch, A, b_is_sparse=measurement_is_sparse, name='y2_batch')
    '''
    
    # define all losses
    if hparams.measurement_type == 'pilot':
        # only polit Loss
        m_loss1_batch =  tf.abs(utils.get_tf_pilot(Rx) - utils.get_tf_pilot(Rx_hat))
        m_loss2_batch =  (utils.get_tf_pilot(Rx) - utils.get_tf_pilot(Rx_hat))**2
        zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
        #d_loss1_batch = -tf.log(prob)
        #d_loss2_batch =  tf.log(1-prob)
    else:
        m_loss1_batch =  tf.reduce_mean(tf.abs(Rx - Rx_hat), 1)
        m_loss2_batch =  tf.reduce_mean((Rx - Rx_hat)**2, 1)
        zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
        #d_loss1_batch = -tf.log(prob)
        #d_loss2_batch =  tf.log(1-prob)
        
    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch 
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)
    #d_loss1 = tf.reduce_mean(d_loss1_batch)
    #d_loss2 = tf.reduce_mean(d_loss2_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    #restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
    restorer_gen.restore(sess, restore_path_gen)
    #restorer_discrim.restore(sess, restore_path_discrim)

    def estimator(Tx_val, Rx_val, Pilot_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)

        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {Tx: Tx_val, Rx: Rx_val, Pilot: Pilot_val}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                print logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val)

            H_hat_val, total_loss_val = sess.run([H_hat, total_loss], feed_dict=feed_dict)
            best_keeper.report(H_hat_val, total_loss_val)
        return best_keeper.get_best()

    return estimator


def vae_estimator(hparams):

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    Tx = tf.placeholder(tf.float32, shape=hparams.image_shape, name='Tx')
    Rx = tf.placeholder(tf.float32, shape=hparams.image_shape, name='Rx')

    # Create the generator
    # TODO: Move z_batch definition here
    z_batch, H_hat, restore_path, restore_dict = channel_model_def.vae_gen(hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        Rx_hat = tf.identity(x_hat_batch, name='y_hat_batch')
    elif hparams.measurement_type == 'pilot':
        Rx_hat = utils.multiComplex(H_hat,Tx);
        # Rx_hat = tf.multiply(H_hat, Tx, name='y_hat')  # TODO complex mult
    else:
        Rx_hat = tf.multiply(H_hat, Tx, name='y_hat') 

    # define all losses
    m_loss1_batch = tf.reduce_mean(tf.reduce_mean(tf.abs(Rx - Rx_hat), 1),0)
    m_loss2_batch = tf.reduce_mean(tf.reduce_mean((Rx - Rx_hat)**2, 1),0)
    zp_loss_batch = tf.reduce_sum(z_batch**2, 1)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams)
    update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(Tx_val, Rx_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        if hparams.measurement_type == 'project':
            feed_dict = {Rx: Rx_val}
        else:
            feed_dict = {Tx: Tx_val, Rx: Rx_val}
        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                print logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val)

            H_hat_val, total_loss_batch_val = sess.run([H_hat, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(H_hat_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator


def learned_estimator(hparams):

    sess = tf.Session()
    y_batch, x_hat_batch, restore_dict = mnist_model_def.end_to_end(hparams)
    restore_path = utils.get_A_restore_path(hparams)

    # Intialize and restore model parameters
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):  # pylint: disable = W0613
        """Function that returns the estimated image"""
        x_hat_batch_val = sess.run(x_hat_batch, feed_dict={y_batch: y_batch_val})
        return x_hat_batch_val

    return estimator
