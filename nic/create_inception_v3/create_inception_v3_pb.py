# -*- coding:utf-8 -*-

import tensorflow as tf
import inception_v3 as iv3

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("ckpt_file", 'ckpt/inception_v3.ckpt', "Inception-v3 checkpoint file.")
tf.flags.DEFINE_string('log_dir', 'logs/', "TensorBoard log directory.")
tf.flags.DEFINE_string('output_dir', './', "Output directory.")
tf.flags.DEFINE_string('output_file', 'inception_v3.pb', "Output file name.")

#　Inception-v3を読み込み
input_img = tf.placeholder(tf.float32,[None, 299, 299, 3], name='input_image')
arg_scope = iv3.inception_v3_arg_scope()
with tf.contrib.slim.arg_scope(arg_scope):
    logits, end_points = iv3.inception_v3(inputs=input_img, is_training=False, num_classes=1001)

#　計算グラフ取得
graph = tf.get_default_graph()

#　TensorBordで確認できるように
writer = tf.summary.FileWriter(FLAGS.log_dir, graph)
writer.close()

#　pb形式で書き出し
tf.train.write_graph(graph, FLAGS.output_dir, FLAGS.output_file)
saver = tf.train.Saver()

with tf.Session() as sess:
    #チェックポイントが読み込めるか念のために確認
    saver.restore(sess, FLAGS.ckpt_file)