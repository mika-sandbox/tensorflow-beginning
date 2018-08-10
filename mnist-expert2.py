# TensorFlow - Deep MNIST for Experts
# https://www.tensorflow.org/get_started/mnist/pros
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST データセットを用意
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Nx784 の入力
x = tf.placeholder(tf.float32, shape=[None, 784])

# Nx10 のラベルデータ
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight の初期化 (0ではなく標準偏差=0.1 となる値で初期化する)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# bias の初期化 (0.1)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込みを行う
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
# strides=[1, 1, 1, 1] : 1x1 ずつ移動させる (index=0, 3 は固定値)
# paddings='SAME'      : 足りない部分を0で埋める
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# プーリングを行う
# https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
# ksize=[1, 2, 2, 1]   : 2x2 の範囲でプーリングを行う
# 以下、 conv2d と同じ
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# =====
# 1st Layer

# 5x5 : 計算範囲サイズ
# 1   : 入力チャンネル数
# 32  : 出力チャンネル数
W_conv1 = weight_variable([5, 5, 1, 32])

# バイアス
b_conv1 = bias_variable([32])

# 画像を 28x28x1 の行列に変換する(height x width x brightness)
# 最初の -1 は現在のサイズを維持する
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 活性化関数として ReLU を使用
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 入力された画像をプーリングする IN:[28x28] -> OUT:[14x14]
h_pool1 = max_pool_2x2(h_conv1)


# =====
# 2nd Layer

# 5x5 : 計算範囲サイズ
# 32  : 入力チャンネル
# 64  : 出力チャンネル
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#プーリングIN:[14x14] -> OUT: [7x7]
h_pool2 = max_pool_2x2(h_conv2)

# =====
# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 7 (pixs) x 7 (pixs) x 64 (features)
b_fc1 = bias_variable([1024])

# 行列をベクトルに変換 (2階 Tensor -> 1階 Tensor)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# =====
# Dropout
# 過学習を防ぐため、いくつかのノードを無効にする
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# =====
# Readout
W_fc2 = weight_variable([1024, 10]) # 1024個の特徴から、 10個の出力にする
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# =====
# Train

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# Adam 法(学習率=0.0001)を使って、 cross_entropy を最小化する
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 評価
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

