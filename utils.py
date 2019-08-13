import tensorflow as tf
import numpy as np

def counter(scope='counter'):
	with tf.variable_scope(scope):
		counter = tf.Variable(0, dtype=tf.int32, name='counter')
		update_cnt = tf.assign(counter, tf.add(counter, 1))
		return counter, update_cnt

def re_normalize(images, means, stds):
    mean = tf.constant(means, dtype=tf.float32, shape=[1, 1, 1, 3])
    std = tf.constant(stds, dtype=tf.float32, shape=[1, 1, 1, 3])

    # res = images * std
    images = images + mean
    return images


# Output: RGB imagessh geh  
def get_visualization(images, illums_est, illums_pooled, illums_ground, BATCH_SIZE):
    #target_shape = (512, 512)
    target_shape = (730, 1096)
    confidence = tf.sqrt(tf.reduce_sum(illums_est ** 2, axis=3, keepdims=True))
    con = tf.split(confidence, BATCH_SIZE, axis=0)
    for i, x in enumerate(con):
        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
        con[i] = x
    vis_confidence = tf.concat(con, axis=0)
    # confidence = tf.sqrt(tf.reduce_sum(illums_est**2, axis=3))
    # vis_confidence = confidence[:, :, :, None]

    vis_est = tf.nn.l2_normalize(illums_est, 3)
    img = images / 255.0
    img = re_normalize(
        img, [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
    raw_images = tf.pow(img, 2.2) * 65535
    img_corrected = tf.pow(
        raw_images / 65535 / illums_pooled[:, None, None, :] /
        tf.reduce_mean(illums_pooled, axis=1, keepdims=True)[:, None, None, :],
        1 / 2.2)
    fcn_padding = 0

    visualization = [
        img, # raw image
        img_corrected, # recovered image
        # vis_confidence_colored,
        vis_confidence,    # confidence map
        vis_confidence * vis_est,       
        vis_est,
        tf.nn.l2_normalize(illums_pooled, 1)[:, None, None, :]   # ground truth color
    ]
    vis_confidence = tf.image.resize_images(
            vis_confidence, (target_shape[0], target_shape[1]),
            method=tf.image.ResizeMethod.AREA)

    for i in range(len(visualization)):
        vis = visualization[i]
        if i == 0:
            padding = 0
        else:
            padding = fcn_padding
        if int(vis.get_shape()[3]) == 1:
            vis = vis * np.array((1, 1, 1)).reshape(1, 1, 1, 3)
        vis = tf.image.resize_images(
            vis, (target_shape[0], target_shape[1]),
            method=tf.image.ResizeMethod.AREA)

        vis = tf.pad(vis,
                tf.constant([[0, 0], [padding, padding], [padding, padding],
                             [0, 0]]))
        vis = tf.pad(vis - 1, tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])) + 1
        visualization[i] = vis
    visualization[3] = visualization[4] * visualization[2]

    visualization_lines = []
    images_per_line = 3
    for i in range(len(visualization) // images_per_line):
        visualization_lines.append(
            tf.concat(
                axis=2,
                values=visualization[i * images_per_line:(i + 1) * images_per_line]))
    visualization = tf.maximum(0.0, tf.concat(axis=1, values=visualization_lines))
    visualization = visualization[:, :, :, ::-1]
    vis_confidence = vis_confidence[:, :, :, ::-1]
    img_corrected = img_corrected[:, :, :, ::-1]
    print('visualization shape', visualization.shape)
    
    # return vis_confidence
    return visualization
    # return visualization[4]
    #return img_corrected
