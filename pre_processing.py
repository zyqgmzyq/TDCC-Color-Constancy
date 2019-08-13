import cv2
import numpy as np
import random
import os
import pandas as pd
import scipy.io
import tensorflow as tf
import re
IMAGE_SHOW = False
# 图像保存为img=raw//2

class GehlerDataSet():
    # get the dataset name
    def get_name(self):
        return 'gehler'

    # get the dataset directory
    def get_directory(self):
        return '/media/Store3/zyq/Dataset/ColorConstancy/' + self.get_name() + '/'
    
    # make csv 
    def make_csv(self):
        filenames = []
        all_labels = []
        PNG_files = os.listdir(os.path.join(self.get_directory(), 'images'))
        PNG_files = [os.path.join(self.get_directory(), 'images', x) for x in PNG_files]
        PNG_files = sorted(PNG_files)
        ground_truth = scipy.io.loadmat(self.get_directory() + 'ground_truth.mat')['real_rgb']
        ground_truth /= np.linalg.norm(ground_truth, axis=1)[..., np.newaxis]
        filenames.extend(PNG_files)
        all_labels.extend(ground_truth)

        zip_all = list(zip(filenames, all_labels))
        random.shuffle(zip_all)
        random.shuffle(zip_all)
        filenames, all_labels = zip(*zip_all)
        pds = pd.DataFrame()
        pds['ImagePath'] = filenames
        pds['Labels'] = all_labels
        pds.to_csv("gehler.csv", index=False)

    # get the coord
    def get_mcc_coord(self, img_path):
        file_path = self.get_directory() + 'coordinates/' + (img_path.split('/')[-1]).split('.')[0] + '_macbeth.txt'
        with open(file_path, 'r') as f:
            lines = f.readlines()
            width, height = map(float, lines[0].split())
            scale_x = 1/width
            scale_y = 1/height
            lines = [lines[1], lines[2], lines[4], lines[3]]
            polyen = []
            for line in lines:
                line = line.strip().split()
                x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
                polyen.append((x, y))
            return np.array(polyen, dtype='float32')

    # load image and remove the black_level
    def load_image(self, img_path):
        img = cv2.imread(img_path, -1)

        # img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = np.array(img, dtype='float32')
        if img_path.__contains__('IMG'):
            black_level = 129
        else:
            black_level = 1
        img = np.maximum(img - black_level, [0, 0, 0])
        return img

    # load img and remove the mcc
    def load_img_without_mcc(self, img_path):
        BOARD_FILL_COLOR = 1e-5
        img = self.load_image(img_path)
        img = (np.clip(img / img.max(), 0, 1)*65535.0).astype(np.uint16)
        polygon = self.get_mcc_coord(img_path)*np.array([img.shape[1], img.shape[0]])
        polygon = polygon.astype(np.int32)
        cv2.fillPoly(img, [polygon], (BOARD_FILL_COLOR,) * 3)
        return img

    # make tf_records
    def write(self, tf_name, img_paths, labels):
        writer = tf.python_io.TFRecordWriter(tf_name)
        label = np.zeros(3, dtype=np.float32)
        for img_path, line in zip(img_paths, labels):
            line = line[1:][:-1]
            line = re.sub(r"\s{2,}", " ", line)
            ll = line.split(', ')
            label[0] = float(ll[0])
            label[1] = float(ll[1])
            label[2] = float(ll[2])
            img = self.load_img_without_mcc(img_path)
            shape = img.shape
            height, width = shape[:2]
            print("before:", shape)
            img = cv2.resize(img,(width // 2, height // 2))
            width = width // 2
            height = height // 2
            print("after", img.shape)
            if IMAGE_SHOW:
                cv2.imshow('img', cv2.resize(
                    np.power(img / 65535., 1.0/2.2), (0, 0),
                    fx=0.25,
                    fy=0.25))
                cv2.waitKey(0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                # 图片对应多个结果
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
            }))
            writer.write(example.SerializeToString())
        writer.close()


class chengDataSet():

    # get the dataset name
    def get_name(self):
        return 'NUS'

    # get the dataset directory
    def get_directory(self):
        return '/media/Store3/zyq/Dataset/' + self.get_name() + '/'


    # make csv
    def make_csv(self):
        camera_list = os.listdir(self.get_directory())
        
        for camera in camera_list:
            filenames = []
            all_labels = []
            darkness = []
            saturation = []
            cc_coordss = []
            PNG_files = os.listdir(os.path.join(
                self.get_directory(), camera, 'PNG'))
            PNG_files = [os.path.join(self.get_directory(), camera,
                                      'PNG', x) for x in PNG_files]
            PNG_files = sorted(PNG_files,
                               key=lambda name: int(name[-8:-4]))
            ground_truth = scipy.io.loadmat(self.get_directory() +
                                            camera + '/'+
                                            camera.replace('JPG', 'gt') +
                                            '.mat')
            illums = ground_truth['groundtruth_illuminants']
            darkness_level = ground_truth['darkness_level']
            saturation_level = ground_truth['saturation_level']
            cc_coords = ground_truth['CC_coords']
            illums /= np.linalg.norm(illums, axis=1)[..., np.newaxis]

            filenames.extend(PNG_files)
            all_labels.extend(illums)
            for x in PNG_files:
                darkness.extend(darkness_level)
                saturation.extend(saturation_level)
            cc_coordss.extend(cc_coords)

            zip_all = list(zip(filenames, all_labels, darkness,
                            saturation, cc_coordss))
            random.shuffle(zip_all)
            random.shuffle(zip_all)
            random.shuffle(zip_all)
            filenames, all_labels, darkness, saturation, cc_coordss= zip(*zip_all)
            pds = pd.DataFrame()
            pds['ImagePath'] = filenames
            pds['Labels'] = all_labels
            pds['Darkness'] = darkness
            pds['Saturation'] = saturation
            pds['Sc_coord'] = cc_coordss
            pds.to_csv("{}_nus.csv".format(camera), index=False)


    # load image and remove the black_level
    def load_image(self, img_path, darkness_level, saturation_level):
        img = cv2.imread(img_path, -1)
        img = np.array(img, dtype='float32')
        img = np.maximum(img - darkness_level, [0, 0, 0])
        img *= 1.0 / saturation_level
        return img

    # load img and remove the mcc
    def load_img_without_mcc(self, img_path, dark_lev, sat_lev, ccord):
        BOARD_FILL_COLOR = 1e-5
        dark_lev = dark_lev[1:][:-2]
        dark_lev = int(dark_lev)

        sat_lev = sat_lev[1:][:-2]
        sat_lev = int(sat_lev)

        cod = ccord[1:][:-1]
        cod = re.sub(r"\s{2,}", " ", cod)
        if cod.startswith(' '):
            cod = cod.split(', ')
            cod = cod[1:]
        else:
            cod = cod.split(', ')
        print(cod)

        y1, y2, x1, x2 = cod[0], cod[1], cod[2], cod[3]
        print(y1, y2, x1, x2)

        img = self.load_image(img_path, dark_lev, sat_lev)
        img = (np.clip(img / img.max(), 0, 1) * 65535.0).astype(np.uint16)

        polygon = np.array([(x1, y1), (x1, y2),
                              (x2, y2), (x2, y1)])
        polygon = polygon.astype(np.int32)
        cv2.fillPoly(img, [polygon], (BOARD_FILL_COLOR,) * 3)
        return img


    def get_dark_sat_cco(self, csv_name):
        df = pd.read_csv(csv_name)
        filenames = df.ImagePath.tolist()
        labels = df.Labels.tolist()
        dark = df.Darkness.tolist()
        sat = df.Saturation.tolist()
        coord = df.Sc_coord.tolist()
        return filenames, labels, dark, sat, coord


    # make tf_records
    def write(self, tf_name, img_paths, labels, dark_lev, sat_lev, ccord):
        writer = tf.python_io.TFRecordWriter(tf_name)
        label = np.zeros(3, dtype=np.float32)
        for img_path, line, dar, s, cood in \
                zip(img_paths, labels, dark_lev, sat_lev, ccord):
            line = line[1:][:-1]
            # line = line[2:][:-1]
            line = re.sub(r"\s{2,}", " ", line)
            ll = line.split(', ')
            label[0] = float(ll[0])
            label[1] = float(ll[1])
            label[2] = float(ll[2])
            print(img_path, label)
            img = self.load_img_without_mcc(img_path, dar, s, cood)
            shape = img.shape
            height, width = shape[:2]
            print('before:', shape)
            img = cv2.resize(img,(width // 2, height // 2))
            shape = img.shape
            print('after:', shape) 
            width = width // 2
            height = height // 2
            print(img.shape)
           
            if IMAGE_SHOW:
                cv2.imshow('img', cv2.resize(
                    np.power(img / 65535., 1.0 / 2.2), (0, 0),
                    fx=0.25,
                    fy=0.25))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                # 图片对应多个结果
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
            }))
            writer.write(example.SerializeToString())
        writer.close()


class CubeDataset():
    def get_name(self):
        return 'cube'
    
    def get_directory(self):
        return '/media/Store3/zyq/Dataset/ColorConstancy/' + self.get_name() + '/'

    def make_csv(self):
        filenames = []
        all_labels = []
        PNG_files = os.listdir(os.path.join(self.get_directory(), 'PNG'))
        PNG_files = [os.path.join(self.get_directory(), 'PNG', x) for x in PNG_files]
        PNG_files = sorted(PNG_files, key=lambda name: int(name[50:-4]))

        with open(self.get_directory() + 'cube_gt.txt', 'r') as file:
            ground_truth = file.readlines()
       
        filenames.extend(PNG_files)
        all_labels.extend(ground_truth)

        zip_all = list(zip(filenames, all_labels))
        random.shuffle(zip_all)
        random.shuffle(zip_all)
        random.shuffle(zip_all)
        filenames, all_labels = zip(*zip_all)
        pds = pd.DataFrame()
        pds['ImagePath'] = filenames
        pds['Labels'] = all_labels
        pds.to_csv("cube.csv", index=False)

    def load_image(self, img_path):
        img = cv2.imread(img_path, -1)
        img = np.array(img, dtype='float32')
        return img

    def load_mask_img(self, img_path):
        img = self.load_image(img_path)
        img = (np.clip(img/img.max(), 0, 1) * 65535.0).astype(np.uint16)
        img[1050:, 2050:, :] = 0
        return img

    def write(self, tf_name, img_paths, labels):
        writer = tf.python_io.TFRecordWriter(tf_name)
        label = np.zeros(3, dtype=np.float32)
        for img_path, line in zip(img_paths, labels):
            line = line[1:][:-1]
            # line = line[2:][:-1]
            line = re.sub(r"\s{2,}", " ", line)
            ll = line.split(' ')
            label[0] = float(ll[0])
            label[1] = float(ll[1])
            label[2] = float(ll[2])
            print(img_path, label)
            img = self.load_mask_img(img_path)
            shape = img.shape
            height, width = shape[:2]
            print('before:', shape)
            img = cv2.resize(img,(width // 2, height // 2))
            shape = img.shape
            print('after:', shape) 
            width = width // 2
            height = height // 2
            print(img.shape)
           
            if IMAGE_SHOW:
                cv2.imshow('img', cv2.resize(
                    np.power(img / 65535., 1.0 / 2.2), (0, 0),
                    fx=0.25,
                    fy=0.25))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                # 图片对应多个结果
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
            }))
            writer.write(example.SerializeToString())
        writer.close()


def create_gehler(num):
    dataset = GehlerDataSet()
    if not os.path.exists('./gehler.csv'):
        dataset.make_csv()
    df = pd.read_csv('./gehler.csv')
    filenames = df.ImagePath.tolist()
    labels = df.Labels.tolist()
    print(len(labels))
    split_size = len(filenames) // num
    for i in range(num):
        start = i * split_size
        end = (i+1) * split_size
        if i==num-1:
            dataset.write('split{}.tfrecords'.format(i + 1),
                          filenames[start:],
                          labels[start:])
        else:
            dataset.write('split{}.tfrecords'.format(i + 1),
                          filenames[start:end],
                          labels[start:end])
    print('size', split_size)


def create_cheng(num):
    dataset = chengDataSet()
    camera_list = os.listdir('/media/Store3/zyq/Dataset/NUS/')
    for camera in camera_list:
        if not os.path.exists('{}_nus.csv'.format(camera)):
            dataset.make_csv()
        filenames, labels, dark, sat, coord = dataset.get_dark_sat_cco('{}_nus.csv'.format(camera))
        print('len', len(labels))
        split_size = len(filenames) // num
        for i in range(num):
            start = i * split_size
            end = (i+1) * split_size
            if i==num-1:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                dataset.write('{}_nus{}.tfrecords'.format(camera, i+1),
                            filenames[start:],
                            labels[start:],
                            dark[start:],
                            sat[start:],
                            coord[start:])
            else:
                dataset.write('{}_nus{}.tfrecords'.format(camera, i+1),
                            filenames[start:end],
                            labels[start:end],
                            dark[start:end],
                            sat[start:end],
                            coord[start:end])
        print('size', split_size)

def create_cube(num):
    dataset = CubeDataset()
    if not os.path.exists('./cube.csv'):
        dataset.make_csv()
    df = pd.read_csv('./cube.csv')
    filenames = df.ImagePath.tolist()
    labels = df.Labels.tolist()
    print(len(labels))
    split_size = len(filenames) // num
    for i in range(num):
        start = i * split_size
        end = (i+1) * split_size
        if i==num-1:
            dataset.write('cube{}.tfrecords'.format(i + 1),
                          filenames[start:],
                          labels[start:])
        else:
            dataset.write('cube{}.tfrecords'.format(i + 1),
                          filenames[start:end],
                          labels[start:end])
    print('size', split_size)


if __name__ == '__main__':
    # create_cheng(3)
    # create_gehler(3)
    create_cube(3)


