#-*- coding:utf-8 -*-

import tensorflow as tf
from skimage import io

def parse(example):

    #TFRecordをパース
    features = tf.parse_single_example(
        example,
        features={
            'data': tf.FixedLenFeature([], dtype=tf.string)
        })
    #バイト列のままになっているので元の画像の形式に変換
    img = features['data']
    img = tf.image.decode_jpeg(img)
    return img


#TFRecordファイルを読み込みパース用の関数を適用
dataset = tf.data.TFRecordDataset(['test.tfrecord']).map(parse)

#データセットを1周するイテレータ
iterator = dataset.make_one_shot_iterator()
#イテレータから要素を取得
next_element = iterator.get_next()

with tf.Session() as sess:
    #データセットから画像を1件取得
    jpeg_img = sess.run(next_element)
    #scikit-imageで表示
    io.imshow(jpeg_img)
    io.show()


