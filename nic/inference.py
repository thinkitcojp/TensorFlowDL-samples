#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from PIL import Image

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('target_img', '', "Target image file path.")
tf.flags.DEFINE_string('dictionary_path', 'data/dictionary.txt', "Dictionary file path.")
tf.flags.DEFINE_string('inference_pb', 'im2txt.pb', "Inference Graph pb file path.")


#辞書ファイルの読み込み
id_to_word = []
with open(FLAGS.dictionary_path, 'r') as f:
    for line in f:
        word = line.rstrip('\n')
        id_to_word.append(word)


# pbファイルを読み込んでデフォルトグラフにロード
with tf.gfile.FastGFile(FLAGS.inference_pb, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# 入力のtf.placeholderを取得
image_placeholder = tf.get_default_graph().get_tensor_by_name('input_img:0')
# 推論結果を取得
caption_ids = tf.get_default_graph().get_tensor_by_name('captioning/decoder/transpose_1:0')

with tf.Session() as sess:

    # 画像データ取得
    img = Image.open(FLAGS.target_img)
    if img.mode != "RGB":
        # 画像がRGB出ない場合は変換しておく
        img = img.convert("RGB")
    # 画像サイズを299×299に、値を0.0-1.0のNumpy配列に変更
    img = [np.asarray(img.resize((299,299)))/255.0]

    features = sess.run(caption_ids, feed_dict={image_placeholder: img})[0]
    #結果を単語に変換
    caption = ''
    for i in features:
        caption += id_to_word[i]

    print(caption)
