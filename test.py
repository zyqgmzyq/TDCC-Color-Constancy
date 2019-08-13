import sys
import tensorflow as tf
import data
import resnet
import os
import cv2 as cv
import numpy as np
import utils
import argparse

BATCH_SIZE = 1
CAPACITY = 2000
is_weight = True
format_data = 'channels_last'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--testdata', dest='testdata', default='split3.tfrecords', help='which dataset to test')
parser.add_argument('--checkpoint', dest='checkpoint', default='./gehler/test1/', help='checkpoint save dir')
parser.add_argument('--restore_dir', dest='restore_dir', default='./gehler/train1/model.ckpt-5999', help='restore dir')
parser.add_argument('--loss_dir', dest='loss_dir', default='./gehler/loss1.txt', help='loss save dir')
parser.add_argument('--visualize', dest='visualize', default='./gehler/test_img1/{:d}_{:d}_{:.3f}.jpg', help='the train visualize img save dir')

args = parser.parse_args()
testdata = args.testdata.split(' ')
checkpoint = args.checkpoint
restore_dir = args.restore_dir
loss_dir = args.loss_dir
visualize = args.visualize
print(testdata, checkpoint, restore_dir, loss_dir, visualize)
list_loss = []


def evaluate(MAX_STEP, tf_file, logs_test_dir, restore_dir, loss_dir, rgb_write_dir):
    filenames = tf.placeholder(tf.string, shape=[None])
    validation_filenames = tf_file
    iterator = data.read_and_decode(filenames, BATCH_SIZE, False)

    sess = tf.Session()
    sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
    test_imgs, test_labels = iterator.get_next()

    test_logits, c5 = resnet.ResNet50(test_imgs, False)
    test_p2, test_p3, test_p4, test_p5 = resnet.FPN(test_logits, False)
    test_est_labels2, test_fea_rgbs2= resnet.cal_loss(test_p2, False, 'p2', is_weight)
    test_est_labels3, test_fea_rgbs3= resnet.cal_loss(test_p3, False, 'p3', is_weight)
    test_est_labels4, test_fea_rgbs4= resnet.cal_loss(test_p4, False, 'p4', is_weight)
    test_est_labels5, test_fea_rgbs5= resnet.cal_loss(test_p5, False, 'p5', is_weight)

    test_est_labels = (test_est_labels2 + test_est_labels3 + test_est_labels4 + test_est_labels5)/4.0
    # test_est_labels, test_fea_rgbs= resnet.cal_loss(test_logits, False, 'logits')
    
    test_losss2 = resnet.losses(test_est_labels2, test_labels)
    test_losss3 = resnet.losses(test_est_labels3, test_labels)
    test_losss4 = resnet.losses(test_est_labels4, test_labels)
    test_losss5 = resnet.losses(test_est_labels5, test_labels)

    visualization2 = utils.get_visualization(test_imgs, test_fea_rgbs2, test_est_labels2, test_labels, BATCH_SIZE)
    visualization3 = utils.get_visualization(test_imgs, test_fea_rgbs3, test_est_labels3, test_labels, BATCH_SIZE)
    visualization4 = utils.get_visualization(test_imgs, test_fea_rgbs4, test_est_labels4, test_labels, BATCH_SIZE)
    visualization5 = utils.get_visualization(test_imgs, test_fea_rgbs5, test_est_labels5, test_labels, BATCH_SIZE)
    

    test_losss = resnet.losses(test_est_labels, test_labels)
    summary_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
    saver = tf.train.Saver()
    saver.restore(sess, restore_dir)
        
    try:
        for step in range(MAX_STEP):
            test_ill_label, test_est_label, test_loss, test_loss2, test_loss3, test_loss4, \
            test_loss5, vis_img2, vis_img3, vis_img4, vis_img5, summary_str = sess.run(
                [test_labels, test_est_labels, test_losss, test_losss2, test_losss3, 
                test_losss4, test_losss5, visualization2, visualization3, \
                visualization4, visualization5, summary_op])
            print('label:', test_ill_label)
            print('estimate:', test_est_label)
            print('Step %d, test loss = %.2f' % (step + 1, test_loss),\
            'loss2 = %.2f loss3 = %.2f loss4 = %.2f loss5 = %.2f' \
            % (test_loss2, test_loss3, test_loss4, test_loss5))
            

            list_loss.append(test_loss)
            
            with open(loss_dir, 'a') as f:
                f.write('%.2f ' % (test_loss))
                f.write('%.2f ' % (test_loss2))
                f.write('%.2f ' % (test_loss3))
                f.write('%.2f ' % (test_loss4))
                f.write('%.2f ' % (test_loss5))
                f.write(str(test_ill_label))
                f.write(str(test_est_label))
                f.write("\n")

            # for k, merged in enumerate(vis_img2):
            #     cv.imwrite(visualize.format(step + 1, k + 4, test_loss2).strip('\r'),
            #                 merged * 255)
            # for k, merged in enumerate(vis_img3):
            #     cv.imwrite(visualize.format(step + 1, k + 3, test_loss3).strip('\r'),
            #                 merged * 255)
            # for k, merged in enumerate(vis_img4):
            #     cv.imwrite(visualize.format(step + 1, k + 2, test_loss4).strip('\r'),
            #                 merged * 255)
            # for k, merged in enumerate(vis_img5):
            #     cv.imwrite(visualize.format(step + 1, k + 1, test_loss5).strip('\r'),
            #                 merged * 255)
            train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_test_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    sess.close()
    metric(list_loss)
    # cv.waitKey(0)


def metric(loss_list):
    loss_list.sort()
    print('------------------------------------')
    print('25:', np.mean(loss_list[:int(0.25 * len(loss_list))]))
    print('med:', np.percentile(loss_list, 50))
    s1 = np.percentile(loss_list, 25)
    s2 = np.percentile(loss_list, 50)
    s3 = np.percentile(loss_list, 75)
    print('tri:', 0.25 * (s1 + s2 + s3))

    print('mean:', np.mean(loss_list))
    print('75:', np.mean(loss_list[int(0.75 * len(loss_list)):]))
    print('95:', np.percentile(loss_list, 95))



def feature(featuremap, x, _tra_img1):
    tra_weight_draw = featuremap[x, :, :, :]
    tra_weight_draw = (tra_weight_draw - tra_weight_draw.min()) / \
                      (tra_weight_draw.max() - tra_weight_draw.min())
    tra_weight_draw *= 255.0
    shape = _tra_img1.shape
    height = shape[0]
    width = shape[1]
    tra_weight_draw = cv.resize(tra_weight_draw, (width, height))
    return tra_weight_draw


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 training.py tf_file1 logs_train_dir1 rgb_write_dir1')
        exit(-1)
    evaluate(200, testdata, checkpoint, restore_dir, loss_dir, visualize)


