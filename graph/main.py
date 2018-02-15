#-*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant(3, name='const1') #定数a
b = tf.Variable(0, name='val1') #変数b

# aとbを足す
add = tf.add(a,b)
#変数bに足した結果をアサイン
assign = tf.assign(b, add)
c = tf.placeholder(tf.int32, name='input') #入力c
# アサインした結果とcを掛け算
mul = tf.multiply(assign, c)
#変数の初期化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #初期化を実行
    sess.run(init)
    for i in range(3):
        # 掛け算を計算するまでのループを実行
        print(sess.run(mul, feed_dict={c:3}))
