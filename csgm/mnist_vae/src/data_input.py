# Get input
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def mnist_data_iteratior():
    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    def iterator(hparams, num_batches):
        for _ in range(num_batches):
            yield mnist.train.next_batch(hparams.batch_size)
    return iterator



def channel_data_iteratior(hparams):
    raw_dataset = tf.data.TFRecordDataset('../train.tfrecords')
    def parse_function(example_proto):
        dics = {'H_data': tf.FixedLenFeature(shape=(1,2048), dtype=tf.float32),
                'Tx_data': tf.FixedLenFeature(shape=(1,2048), dtype=tf.float32),
                'Rx_data': tf.FixedLenFeature(shape=(1,2048), dtype=tf.float32)}
        parsed_example = tf.parse_single_example(example_proto, dics)
        return parsed_example
    new_dataset = raw_dataset.map(parse_function)
    shuffle_dataset = new_dataset.shuffle(buffer_size=100000)
    repeat_dataset = shuffle_dataset.repeat(hparams.training_epochs)
    batch_dataset = repeat_dataset.batch(hparams.batch_size)
    iterator = batch_dataset.make_one_shot_iterator()
    return iterator
