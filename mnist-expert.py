# TensorFlow - Deep MNIST for Experts
# https://www.tensorflow.org/get_started/mnist/pros
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST データセットを用意
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# セッションを起動
sess = tf.InteractiveSession()


# =====
# Softmax Regression Model を作成

# 受け取るデータ数は不明のため、行数は任意の数 = None となる
x = tf.placeholder(tf.float32, shape=[None, 784]) # 28x28 = 784
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # 0 to 9 = 10 classifier

# ゼロ埋めしたパラメータ
W = tf.Variable(tf.zeros([784, 10])) # Weight: 784次元の入力を受け取り、10次元の出力を行いたい
b = tf.Variable(tf.zeros([10]))      # bias  : 10次元の出力のため

sess.run(tf.global_variables_initializer())

# y = x .* W + b
y = tf.matmul(x, W) + b

# 実際の値との誤差を計算, この値を最小化するように計算が行われる
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# =====
# Train

# 学習率0.5で学習させる
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 1000回訓練する鵜
for _ in range(1000):
    batch = mnist.train.next_batch(100) # MNIST データからランダムに100個もってくる
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})


# 結果を評価 (tf.argmax で最も確率が高い値のインデックスを取得し、比較する)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 正解率をとる (tf.cast で [True, False, True, True] を [1, 0, 1, 1] にキャスとした後、平均を計算)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

