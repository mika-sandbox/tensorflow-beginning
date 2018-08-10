# TensorFlow - MNIST For ML Beginners
# https://www.tensorflow.org/get_started/mnist/beginners
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST データセットを用意
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 入力は 28x28 の 784 ベクトル
x = tf.placeholder(tf.float32, [None, 784])

# ゼロ埋めした重み (weight), 784x10 の行列
W = tf.Variable(tf.zeros([784, 10]))

# バイアス (bias), 出力値に足される, 1x10 の行列
b = tf.Variable(tf.zeros([10]))

# y = softmax(x * W + b), matmul は行列のかけ算を行う
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正解データに対する CrossEntropy (誤差)を求める, 1x10 の行列
y_ = tf.placeholder(tf.float32, [None, 10])

# CrossEntropy = - \sum{y' * log(y)}
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 学習率 0.05 の最急降下法を用いて、 cross_entropy の最小化を行う
# このとき、バックプロパゲーションも同時に行ってくれている
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# セッションを起動
sess = tf.InteractiveSession()

# Variable 初期化
tf.global_variables_initializer().run()

# 1000 回訓練する
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # MNIST データからランダムに100個とってくる
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 学習


# 学習結果を評価
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 正解率をとる (correct_prediction は boolean なので、True -> 1, False -> 0 として変換して平均をとる)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

