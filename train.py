import sys
import tensorflow as tf
import data
import resnet
import os
import cv2
import time
import argparse
import utils

format_data = 'channels_last'

BATCH_SIZE = 16
CAPACITY = 2000
END_EPOCH = 6000
train_num = 380
is_weight = True
# learning_rate = 0.00001

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='split1.tfrecords,split2.tfrecords', help='which dataset to use')
parser.add_argument('--checkpoint', dest='checkpoint', default='./gehler/train1/', help='checkpoint save dir')
parser.add_argument('--visualize', dest='visualize', default='./gehler/rgb1/{:d}_{:d}_{:.3f}.jpg', help='the train visualize img save dir')

args = parser.parse_args()
dataset = args.dataset.split(',')
checkpoint = args.checkpoint
visualize = args.visualize
print(dataset, checkpoint, visualize)

# counter
it_cnt, update_cnt = utils.counter()

def train(tf_file, checkpoint, visualize):
    restore_dir = checkpoint
    learning_rate = tf.Variable(0.0001, trainable=False)
    decay_lr = tf.assign(learning_rate, tf.maximum(learning_rate / 10.0, 1e-6))
    filenames = tf.placeholder(tf.string, shape=[None])
    training_filenames = tf_file
    with tf.device('/cpu:0'):
        iterator = data.read_and_decode(filenames, BATCH_SIZE, True)
    sess = tf.Session()
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    tra_imgs, tra_labels = iterator.get_next()
    
    logits, tra_c5 = resnet.ResNet50(tra_imgs, True)
    
    p2, p3, p4, p5 = resnet.FPN(logits, True)
    tra_est_labels2, tra_fea_rgbs2= resnet.cal_loss(p2, True, 'p2', is_weight)
    tra_est_labels3, tra_fea_rgbs3= resnet.cal_loss(p3, True, 'p3', is_weight)
    tra_est_labels4, tra_fea_rgbs4= resnet.cal_loss(p4, True, 'p4', is_weight)
    tra_est_labels5, tra_fea_rgbs5= resnet.cal_loss(p5, True, 'p5', is_weight)

    tra_losss2 = resnet.losses(tra_est_labels2, tra_labels)
    tra_losss3 = resnet.losses(tra_est_labels3, tra_labels)
    tra_losss4 = resnet.losses(tra_est_labels4, tra_labels)
    tra_losss5 = resnet.losses(tra_est_labels5, tra_labels)
    tra_losss = (tra_losss2 + tra_losss3 + tra_losss4 + tra_losss5)/4.0


    tra_est_labels = (tra_est_labels2 + tra_est_labels3 + tra_est_labels4 + tra_est_labels5) / 4.0
    tra_fea_rgbs = tra_fea_rgbs2
    # tra_est_labels, tra_fea_rgbs= resnet.cal_loss(logits, True, 'logits')
    
    # tra_losss = resnet.losses(tra_est_labels, tra_labels)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'resnet_model'))
    
    train_op = resnet.trainning(tra_losss, learning_rate)

    visualization = utils.get_visualization(tra_imgs, tra_fea_rgbs, tra_est_labels, tra_labels, BATCH_SIZE)


    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoint, sess.graph)
    try:
        load_checkpoint(restore_dir, sess)
    except:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './resnet_imagenet_v2/model.ckpt-225207')
        print("restore succed!")
    saver = tf.train.Saver()
    
    try:
        for epoch in range(sess.run(it_cnt), END_EPOCH):
            sess.run(update_cnt)
            batches_per_epoch = train_num // BATCH_SIZE
            start_time = time.time()
            for j in range(batches_per_epoch):
                _, lr_value, tra_loss, vis_img = sess.run(
                    [train_op, learning_rate, tra_losss, visualization])

            duration_time = time.time() - start_time
            if epoch % 10 == 0:
                # tra_weight.imshow()
                print('Step %d, train loss = %.2f' % (epoch, tra_loss))
                print("{} images per secs, lr = {}".format(batches_per_epoch * BATCH_SIZE / duration_time, lr_value))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, epoch)

            # if epoch % 200 == 0:
            #     for k, merged in enumerate(vis_img):
            #         if k % 8==0:
            #             cv2.imwrite(visualize.format(epoch + 1, k + 1, tra_loss).strip('\r'),
            #                         merged * 255)
            if (epoch + 1) % 2000 == 0 or (epoch + 1) == END_EPOCH:
                print("start decay lr")
                sess.run(decay_lr)
                checkpoint_path = os.path.join(checkpoint, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    sess.close()


def load_checkpoint(ckpt_dir_or_file, session, var_list=None):
    """Load checkpoint.

    Note:
        This function add some useless ops to the graph. It is better
        to use tf.train.init_from_checkpoint(...).
    """
    print(' [*] Loading checkpoint...')
    if os.path.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    restorer = tf.train.Saver(var_list)
    restorer.restore(session, ckpt_dir_or_file)
    print(' [*] Loading succeeds! Copy variables from % s' % ckpt_dir_or_file)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 training.py tf_file1 checkpoint1 rgb_write_dir')
        exit(-1)

    train(dataset, checkpoint, visualize)
    


