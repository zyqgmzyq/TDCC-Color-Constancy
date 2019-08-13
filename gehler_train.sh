#!/bin/bash
echo "start"
# python3 training.py tf_file1 logs_train_dir1 rgb_write_dir1
# python3 evalAlexnet_test.py tf_file1 logs_test_dir1 restore_dir1 loss_dir1 rgb_test_dir1
# python3 training.py tf_file2 logs_train_dir2 rgb_write_dir2
# python3 evalAlexnet_test.py tf_file2 logs_test_dir2 restore_dir2 loss_dir2 rgb_test_dir2
# python3 training.py tf_file3 logs_train_dir3 rgb_write_dir3
# python3 evalAlexnet_test.py tf_file3 logs_test_dir3 restore_dir3 loss_dir3 rgb_test_dir3
python3 train.py --dataset='split1.tfrecords,split2.tfrecords' --checkpoint='./gehler/train1/' --visualize='./gehler/rgb1/{:d}_{:d}_{:.3f}.jpg'
python3 test.py --testdata='split3.tfrecords' --checkpoint='./gehler/test1/' --restore_dir='./gehler/train1/model.ckpt-5999' --loss_dir='./gehler/loss1.txt' --visualize='./gehler/test_img1/{:d}_{:d}_{:.3f}.jpg'
python3 train.py --dataset='split1.tfrecords,split3.tfrecords' --checkpoint='./gehler/train2/' --visualize='./gehler/rgb2/{:d}_{:d}_{:.3f}.jpg'
python3 test.py --testdata='split2.tfrecords' --checkpoint='./gehler/test2/' --restore_dir='./gehler/train2/model.ckpt-5999' --loss_dir='./gehler/loss2.txt' --visualize='./gehler/test_img2/{:d}_{:d}_{:.3f}.jpg'
python3 train.py --dataset='split2.tfrecords,split3.tfrecords' --checkpoint='./gehler/train3/' --visualize='./gehler/rgb3/{:d}_{:d}_{:.3f}.jpg'
python3 test.py --testdata='split1.tfrecords' --checkpoint='./gehler/test3/' --restore_dir='./gehler/train3/model.ckpt-5999' --loss_dir='./gehler/loss3.txt' --visualize='./gehler/test_img3/{:d}_{:d}_{:.3f}.jpg'
