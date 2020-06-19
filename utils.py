import os
import errno
import numpy as np
import scipy
import scipy.misc
import h5py
import scipy.io as scio
from scipy.ndimage.interpolation import zoom
import tensorflow as tf

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class Channel(object):
    def __init__(self, image_path, epoch, batch_size):

        self.dataname = "channel"
        self.channel = 2
        self.epoch = epoch
        self.image_path = image_path
        self.batch_size = batch_size
        #self.data_iter = self.get_channel_iter(image_path, data_size, batch_size)
        #self.image_list = self.load_channel(image_path=image_path,data_size=data_size)
        
    def get_channel_iter(self):
        dataset = tf.data.TFRecordDataset(self.image_path)
        def parse_function(example_proto):
            dics = {'H_data': tf.FixedLenFeature(shape=(8192,1), dtype=tf.float32),
                    'Tx_data': tf.FixedLenFeature(shape=(8192,1), dtype=tf.float32),
                    'Rx_data': tf.FixedLenFeature(shape=(8192,1), dtype=tf.float32)}
            parsed_example = tf.parse_single_example(example_proto, dics)
            return parsed_example
        dataset = dataset.shuffle(buffer_size=20000)
        dataset = dataset.map(parse_function, num_parallel_calls=8)
        dataset = dataset.repeat(self.epoch+1)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.batch_size*10)
        iterator = dataset.make_one_shot_iterator()
        return iterator
        
'''
    def load_channel(self, image_path, data_size):

        # get the list of image path
        images_list = read_mat_list(image_path, data_size)
        if len(images_list) != data_size:
            print("data_size Err")
        # get the data array of image
        return images_list

    def getShapeForData(self, filenames, resize_w=64):
        array = [get_mat(batch_file, resize_w) for batch_file in filenames]

        sample_images = np.array(array)
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64):
        if batch_num  == 0:

            length = len(self.image_list)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.image_list = np.array(self.image_list)
            self.image_list = self.image_list[perm]

            print ("images shuffle")

        return self.image_list[batch_num * batch_size: (batch_num + 1) * batch_size]
'''

class CelebA(object):
    def __init__(self, image_path):

        self.dataname = "CelebA"
        self.channel = 3
        self.image_list = self.load_celebA(image_path=image_path)

    def load_celebA(self, image_path):

        # get the list of image path
        images_list = read_image_list(image_path)
        # get the data array of image

        return images_list

    def getShapeForData(self, filenames, resize_w=64):
        array = [get_image(batch_file, 128, is_crop=True, resize_w=resize_w,
                           is_grayscale=False) for batch_file in filenames]

        sample_images = np.array(array)
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64):
        ro_num = len(self.image_list) / batch_size - 1
        if batch_num % ro_num == 0:

            length = len(self.image_list)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.image_list = np.array(self.image_list)
            self.image_list = self.image_list[perm]

            print ("images shuffle")

        return self.image_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

class CelebA_HQ(object):
    def __init__(self, image_path):
        self.dataname = "CelebA_HQ"
        resolution = ['data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32', 'data64x64', \
                      'data128x128', 'data256x256', 'data512x512', 'data1024x1024']
        self.channel = 3
        self.image_list = self.load_celeba_hq(image_path=image_path)
        self._base_key = 'data'
        self._len = {k: len(self.image_list[k]) for k in resolution}

    def load_celeba_hq(self, image_path):
        # get the list of image path
        images_list = h5py.File(os.path.join(image_path, "celebA_hq"), 'r')
        # get the data array of image
        return images_list

    def getNextBatch(self, batch_size=64, resize_w=64):
        key = self._base_key + '{}x{}'.format(resize_w, resize_w)
        idx = np.random.randint(self._len[key], size=batch_size)
        batch_x = np.array([self.image_list[key][i] / 127.5 - 1.0 for i in idx], dtype=np.float32)

        return batch_x

def get_mat(image_path, resize_w=64):
    H_data = scio.loadmat(image_path)['H_data']
    H_data = zoom(H_data, zoom=[resize_w/64.0, resize_w/64.0, 1])
    return H_data
    
def get_image(image_path , image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def get_image_dat(image_path , image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread_dat(image_path , is_grayscale), image_size, is_crop , resize_w)

def transform(image, npx=64 , is_crop=False, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])

    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    rate = np.random.uniform(0, 1, size=1)

    if rate < 0.5:
        x = np.fliplr(x)

    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])

        # return scipy.misc.imresize(x[20:218 - 20, 0: 178], [resize_w, resize_w])

    # return scipy.misc.imresize(x[45: 45 + 128, 25:25 + 128], [resize_w, resize_w])

def save_mats(images, size, image_path):
    scio.savemat(image_path, {'H_hat':images})

def save_mat_images(images, size, image_path):
    H_matrix=  images[:,:,:,0]*images[:,:,:,0]+images[:,:,:,1]*images[:,:,:,1]
    return scipy.misc.imsave(image_path, H_matrix[0,:,:])
    
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imread_dat(path, is_grayscale):
    return np.load(path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image
    return img

def inverse_transform(image):
    return ((image + 1.)* 127.5).astype(np.uint8)

def read_image_list(category):
    filenames = []
    print("list file")
    list = os.listdir(category)
    list.sort()
    for file in list:
        if 'jpg' in file:
            filenames.append(category + "/" + file)
    print("list file ending!")
    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames

def read_mat_list(category, file_length):
    filenames = []
    print("list file")
    for i in range(file_length):
        filenames.append(category + "/perfect/H/H_%06d_gt.mat" % (i+1))
    
    print("list file ending!")

    return filenames



