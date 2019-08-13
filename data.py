import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FCN_INPUT_SIZE = 512


def normalize(image, means, stds):
    """normalize a image by substract mean and std
    Args :
        image: 3-D image
        means: list of C mean values
        stds: list of C std values
    Returns:
        3-D normalized image

    for imagenet pretrained model:
        means = [0.485, 0.456, 0.406] R G B
        std = [0.229, 0.224, 0.225] R G B
    """
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
        # channels[i] /= stds[i]
    return tf.concat(axis=2, values=channels)


# 读取tf
def read_and_decode(filenames, batch_size, is_training):
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        features = {'label': tf.FixedLenFeature([3], tf.float32),
                    'img_raw': tf.FixedLenFeature([], tf.string),
                    'width': tf.FixedLenFeature([],tf.int64),
                    'height': tf.FixedLenFeature([], tf.int64)}
        parsed = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed["img_raw"], tf.uint16)
        img_width = tf.cast(parsed['width'], tf.int32)
        img_height = tf.cast(parsed['height'], tf.int32)
        image = tf.reshape(image, [img_height, img_width, 3])
        image = tf.cast(image, tf.float32)
        # labels
        label = tf.cast(parsed["label"], tf.float32)
        if is_training:
            image, label = data_augumentation(image, label)
        # else:
        #     image = tf.image.resize_images(
        #         image,
        #         [img_height // 2, img_width // 2])
        # normalize
        image = tf.pow(image / 65535., 1/2.2)
        image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image = image * 255
        label = tf.nn.l2_normalize(label)
        return image, label

    # Parse the record into tensors.
    dataset = dataset.map(parser, num_parallel_calls=16)
    if is_training:
        # Repeat the input indefinitely.
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    data = tf.contrib.data.prefetch_to_device('/gpu:0')
    iterator = dataset.make_initializable_iterator()
    return iterator


# data_augumentation
def data_augumentation(image, label):
    angle = tf.random_uniform([], minval=-30, maxval=30, dtype=tf.float32)
    scale = tf.random_uniform([], minval=0.1, maxval=1.0, dtype=tf.float32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    min_w = tf.minimum(height, width)
    s = scale * tf.cast(min_w, tf.float32)
    s = tf.cast(tf.round(s), tf.int32)
    s = tf.minimum(tf.maximum(s, tf.constant(10)), min_w)
    start_x = tf.random_uniform([], minval=0, maxval=height - s + tf.constant(1), dtype=tf.int32)
    start_y = tf.random_uniform([], minval=0, maxval=width - s + tf.constant(1), dtype=tf.int32)
    color_aug = tf.random_uniform(shape=[1, 1, 3], minval=0.8, maxval=1.2, dtype=tf.float32)

    def crop(img, label):
        if img is None:
           return None
        img = img[start_x:start_x + s, start_y:start_y + s]
        # rotateds
        img = tf.contrib.image.rotate(img, angle)
        # random crop
        # img = tf.random_crop(image, [FCN_INPUT_SIZE, FCN_INPUT_SIZE, 3])
        # img = tf.image.central_crop(img, 0.8)
        # resize image
        img = tf.image.resize_images(img, [FCN_INPUT_SIZE, FCN_INPUT_SIZE])
        # random flip left right
        img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, tf.float32)
        new_illum = label * color_aug[0][0]
        new_image = img * color_aug
        # new_illum = label
        # new_image = img
        new_image = tf.clip_by_value(new_image, 0, 65535)
        new_illum = tf.clip_by_value(new_illum, 0.01, 100)
        return new_image, new_illum

    return crop(image, label)
