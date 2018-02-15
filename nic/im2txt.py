# -*- coding:utf-8 -*-

import glob
import tensorflow as tf
from PIL import Image
import numpy as np 


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('tfrecord_dir', 'data/tfrecords/', "TFRecords' directory.")
tf.flags.DEFINE_string('dictionary_path', 'data/dictionary.txt', "Dictionary file path.")
tf.flags.DEFINE_string('inference_pb', 'data/im2txt.pb', "Inference Graph pb file path.")
tf.flags.DEFINE_string('img_embedding_pb', 'create_inception_v3/inception_v3_freezed.pb',"Image embedding network pb file path.")
tf.flags.DEFINE_string('model_dir', 'ckpt/', "Saved checkpoint directory.")
tf.flags.DEFINE_string('log_dir', 'logs/', "TensorBoard log directory.")
tf.flags.DEFINE_string('test_img_dir','data/img/for_eval/' ,"Test image directory.")
tf.flags.DEFINE_integer('max_step', 100000, "Num of max train step.")
tf.flags.DEFINE_integer('eval_interval', 100, "Step interval of evaluation.")
tf.flags.DEFINE_integer('embedding_size', 512, "Num of embedded feature size.")
tf.flags.DEFINE_integer('batch_size', 64, "Num of batch size.")
tf.flags.DEFINE_float('learning_rate', 0.0005, "Learning rate.")
tf.flags.DEFINE_float('max_gradient_norm', 5.0, "Max norm of gradient.")

EOS_ID = 0
SOS_ID = 1

def _process_img(encoded_img):

    #画像のデコード [height, width, channel]の3階テンソルになる
    img = tf.image.decode_jpeg(encoded_img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    #サイズが299*299になるようにリサイズ、クロップ
    img = tf.image.resize_images(img, [373, 373])
    img = tf.image.central_crop(img, 0.8)
    return img

def _parse_function(sequence_expample_proto):

    #tf.train.SequenceExampleをパースするメソッド
    context, feature_lists = tf.parse_single_sequence_example(
        sequence_expample_proto,
        context_features = {
            "img_id": tf.FixedLenFeature([], dtype=tf.int64),
            "data": tf.FixedLenFeature([], dtype=tf.string),
        },
        sequence_features = {
            "caption": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })
    #パース結果を格納、画像は加工して格納
    img = _process_img(context['data'])
    caption = feature_lists['caption']

    #正解データのサイズを取得、正解データは<SOS>がないので1を引く
    lengths = tf.size(caption) - 1
    #訓練時decoderへの入力は<EOS>がない
    decoder_input = caption[:-1]
    #正解データは<SOS>がない
    correct = caption[1:]
    
    return (img, lengths, decoder_input, correct)



def build_input(pattern, repeat_count=None ,shuffle=True):

    #訓練用、バリデーション用のtfrecordsファイル名一覧を取得
    files = []
    for file_path in glob.glob(FLAGS.tfrecord_dir+pattern):
        files.append(file_path)

    #TFRecordをパースしてDatasetを作成
    dataset = tf.data.TFRecordDataset(files).map(_parse_function).repeat(count=repeat_count)

    #1000件ずつバッファを取りながらデータをシャッフル
    if shuffle:
        dataset = dataset.shuffle(1000)
    #キャプションデータ長の不均衡を0でpaddingしてミニバッチを作成
    padded_shapes = (tf.TensorShape([300,300,3]), tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None]))
    dataset = dataset.padded_batch(FLAGS.batch_size, padded_shapes=padded_shapes)

    #repeatで指定した回数までループするiteratorを作成
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    return next_element

def build_img_embedding(img_input, embedding_size):

    #画像が崩れていないかTensorBoardで確認
    tf.summary.image('input', img_input*256, max_outputs=10)

    #inception-v3の訓練済みモジュールを読み込み
    with tf.gfile.FastGFile(FLAGS.img_embedding_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    img_embedding = tf.import_graph_def(graph_def, input_map={'input_image':img_input}, return_elements=['InceptionV3/Predictions/Reshape:0'])

    #RNNの中間層に合わせるための全結合
    img_embedding = tf.layers.dense(img_embedding[0], FLAGS.embedding_size)

    return img_embedding


def build_caption(img_embedding, vocab_size, is_train=True, decoder_input=None, decoder_lengths=None, end_token=EOS_ID):

    with tf.name_scope("captioning"):

        batch_size = tf.shape(img_embedding)[0]
        embedding_size = FLAGS.embedding_size
        #単語組み込み用の重みを定義
        word_embedding = tf.get_variable("embeddings", [vocab_size , embedding_size])

        if is_train:
            #decoder_inputからlookup
            embedded_input = tf.nn.embedding_lookup(word_embedding, decoder_input)
            #decoder側への組み込み入力、有効時間長を引数に訓練用のヘルパーを定義
            helper = tf.contrib.seq2seq.TrainingHelper(embedded_input, decoder_lengths)
        else:
            #出力層で最も確率が高いものだけを選択し次の入力にするヘルパーを定義
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embedding, tf.fill([batch_size], SOS_ID), end_token)

        #LSTMのCellを定義
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)
        #画像組み込みベクトルを入力にしてRNNを1ステップ計算してキャプション生成の初期値獲得
        _, initial_state = rnn_cell(img_embedding, rnn_cell.zero_state(batch_size, dtype=tf.float32))

        #出力そうの挙動を定義
        projection_layer = tf.layers.Dense(vocab_size, use_bias=False)
        #デコーダー側の挙動を確定
        decoder = tf.contrib.seq2seq.BasicDecoder(rnn_cell, helper, initial_state, projection_layer)
        #Dynamicデコード
        output, final_state, lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=100)

    return output, lengths

#誤差計測
def build_loss(logit, correct, lengths):
    with tf.name_scope('loss'):
        #有効でない時系列部分をマスク
        max_len = tf.reduce_max(lengths)
        weight = tf.sequence_mask(lengths, max_len, logit.dtype)
        loss = tf.contrib.seq2seq.sequence_loss(logit, correct, weight)
    return loss

#訓練実施
def build_train(loss):
    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        #勾配クリッピング
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        #訓練
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        update = optimizer.apply_gradients(zip(clipped_gradients, params),global_step=global_step)

    return update, global_step

#ckptからのrestoreもしくは初期化
def initialize_model(saver, sess, initializer=None):

    #最新のckptを取得
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt_state:
        last_model = ckpt_state.model_checkpoint_path
        saver.restore(sess,last_model)
    else:
        sess.run(initializer)

def main(argv):


    dataset_dir = FLAGS.tfrecord_dir
    embedding_size = FLAGS.embedding_size

    #辞書ファイルの読み込み
    print("Load dictionary.")
    id_to_word = []
    with open(FLAGS.dictionary_path, 'r') as f:
        for line in f:
            word = line.rstrip('\n')
            id_to_word.append(word)
    vocab_size = len(id_to_word)

    print("Start to build model.")

    train_graph = tf.Graph() #訓練用グラフ
    infer_graph = tf.Graph() #推論用グラフ

    #推論用のグラフ
    with infer_graph.as_default():
        #推論の入力はplaceholder形式
        input_img = tf.placeholder(tf.float32,[None, 299, 299, 3], name='input_img')
        infer_embedding = build_img_embedding(input_img, embedding_size)
        infer_output, _ = build_caption(infer_embedding, vocab_size, is_train=False)

        infer_saver = tf.train.Saver()


    #訓練用のグラフ
    with train_graph.as_default():
        #訓練の入力はDataSetAPIを用いたパイプ
        img , lengths, decoder_input, correct = build_input('train-*')
        #画像組み込み
        train_embedding = build_img_embedding(img, embedding_size)
        #キャプション生成
        train_output, train_lengths = build_caption(train_embedding, vocab_size, decoder_input=decoder_input, decoder_lengths=lengths)
        #誤差計算
        loss = build_loss(train_output.rnn_output, correct, lengths)
        #訓練
        update, global_step = build_train(loss)

        #生成テキストのlogging
        text_ph = tf.placeholder(tf.string, shape=(None,), name='generated')
        tf.summary.text('text_summary', text_ph)
        #ログのマージ
        summary_op = tf.summary.merge_all()
        initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()



    print("Finish building models.")

    #グラフごとにSessionを定義
    sess_infer = tf.Session(graph=infer_graph)
    sess_train = tf.Session(graph=train_graph)

    #TensorBoardのwriterを定義
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, train_graph)

    #inferenceをpb形式でエクスポート
    tf.train.write_graph(infer_graph.as_graph_def(), './data', 'infer_graph.pb', as_text=False)

    #初期化
    initialize_model(train_saver, sess_train, initializer)

    last_step = sess_train.run(global_step)
    print("start training!")
    for i in range(FLAGS.max_step):
        step = last_step + i + 1

        sess_train.run(update)

        #一定ステップごとにモデル保存とloggingを行い文章生成を行う
        if step % FLAGS.eval_interval == 0:
            print("Step%d. Save model."%step)
            #モデル保存
            train_saver.save(sess_train, FLAGS.model_dir+'mymodel')

            #inference用のgraphにckptをロード
            print("Restore to inference graph.")
            initialize_model(infer_saver, sess_infer)

            #チェック用の画像ファイル一覧を取得
            eval_images = []
            for file_path in glob.glob(FLAGS.test_img_dir+'*.jpg'):
                eval_images.append(file_path)

            #画像をモデルの入力に変換
            inference_input = []
            for file in eval_images:
                _img = Image.open(file)
                inference_input.append(np.asarray(_img.resize((299,299)))/255.0)

            #実行
            infer_result = sess_infer.run(infer_output.sample_id, feed_dict={input_img: inference_input})
            #結果を辞書で変換
            captions = []
            for result in infer_result:
                caption = ''
                for j in result:
                    caption += id_to_word[j]
                captions.append(caption)

            #TensorBoardのlogging
            run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_meta = tf.RunMetadata()
            summary_val = sess_train.run(summary_op, feed_dict={text_ph:captions}, options=run_opt, run_metadata=run_meta)
            summary_writer.add_summary(summary_val, step)
            summary_writer.add_run_metadata(run_meta,'step%d'%step)


    summary_writer.close()
    sess_infer.close()
    sess_train.close()

if __name__ == '__main__':
    tf.app.run()