import tensorflow as tf

from utils import mkdir_p
from PGGAN import PGGAN
from utils import CelebA, CelebA_HQ, Channel
flags = tf.app.flags
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string("OPER_NAME", "channel_64_16", "the name of experiments")
flags.DEFINE_string("data_path" , "/home/zby/datasets/channel/data_5.22/train_5.22.tfrecords", "Path of training data, for example /home/hehe/")
flags.DEFINE_string("dataset" , 'channel', "Path of training data, for example /home/hehe/")
flags.DEFINE_integer("data_size", 50000, "dataset size")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("sample_size", 128, "Size of sample")
flags.DEFINE_integer("epoch", 3, "Size of sample")
flags.DEFINE_float("scale_factor", 2.5, "Size of sample")
flags.DEFINE_integer("max_iters", 800, "Maxmization of training number")
flags.DEFINE_float("learn_rate", 0.001, "Learning rate for G and D networks")
flags.DEFINE_integer("lam_gp", 10, "Weight of gradient penalty term")
flags.DEFINE_float("lam_eps", 0.001, "Weight for the epsilon term")
flags.DEFINE_integer("pg", 9, "FLAG of gan training process")
flags.DEFINE_boolean("use_wscale", True, "Using the scale of weight")
flags.DEFINE_boolean("celeba", True, "Whether using celeba or using CelebA-HQ")
FLAGS = flags.FLAGS


if __name__ == "__main__":

    root_log_dir = "./output/{}/logs/".format(FLAGS.OPER_NAME)
    mkdir_p(root_log_dir)

    
    fl = [1,2,2,3,3,4,4,5,5,6,6]
    r_fl = [1,1,2,2,3,3,4,4,5,5,6]

    for i in range(FLAGS.pg):
        
        if FLAGS.dataset == 'celeba':
            data_In = CelebA(FLAGS.data_path)
        elif FLAGS.dataset == 'celeba_hq':
            data_In = CelebA_HQ(FLAGS.data_path)
        elif FLAGS.dataset == 'channel':
            data_In = Channel(FLAGS.data_path, FLAGS.epoch, FLAGS.batch_size)
        

        t = False if (i % 2 == 0) else True
        pggan_checkpoint_dir_write = "./output/{}/model_pggan_{}/{}/".format(FLAGS.OPER_NAME, FLAGS.dataset, fl[i])
        sample_path = "./output/{}/sample_{}_{}".format(FLAGS.OPER_NAME, FLAGS.dataset, fl[i], t)
        mkdir_p(pggan_checkpoint_dir_write)
        mkdir_p(sample_path)
        pggan_checkpoint_dir_read = "./output/{}/model_pggan_{}/{}/".format(FLAGS.OPER_NAME, FLAGS.dataset, r_fl[i])

        pggan = PGGAN(batch_size=FLAGS.batch_size, data_size=FLAGS.data_size, epoch=FLAGS.epoch,
                      model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                      data=data_In, sample_size=FLAGS.sample_size,
                      sample_path=sample_path, log_dir=root_log_dir, learn_rate=FLAGS.learn_rate, lam_gp=FLAGS.lam_gp, lam_eps=FLAGS.lam_eps, PG= fl[i],
                      t=t, use_wscale=FLAGS.use_wscale, scale_factor=FLAGS.scale_factor)

        pggan.build_model_PGGan()
        pggan.train()











