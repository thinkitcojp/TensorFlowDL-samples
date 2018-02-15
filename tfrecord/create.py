#-*- coding:utf-8 -*-

import glob
import tensorflow as tf


#jpegファイルのパスを取得
img_list = [i for i in glob.glob('img/*.jpg')]

with tf.python_io.TFRecordWriter('test.tfrecord') as w:

    for img in img_list:

        #ファイルをバイナリとして読み込み
        with tf.gfile.FastGFile(img, 'rb') as f:
            data = f.read()

        #取得したbyte列をkey,valueに登録
        features = tf.train.Features(feature={
            'data':tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
        })

        #Exampleクラスにkey, valueを登録して書き込み
        example = tf.train.Example(features=features)
        w.write(example.SerializeToString())