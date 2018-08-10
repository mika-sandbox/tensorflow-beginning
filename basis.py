import tensorflow as tf

def run(node, param=None):
    if param is None:
        print(sess.run(node))
    else:
        print(sess.run(node, param))

sess = tf.Session()

# ====================
# 定数
const_node_1 = tf.constant(3.0, dtype=tf.float32)
run(const_node_1)

# ====================
# 足し算
const_node_2 = tf.constant(4.0, dtype=tf.float32)
adder_node_1 = tf.add(const_node_1, const_node_2)
run(adder_node_1)

adder_node_2 = const_node_1 + const_node_2 # こっちの方が直感的
run(adder_node_2)

# ====================
# かけ算
run(const_node_1 * 3)

# ====================
# プレースホルダー
placeholder_a = tf.placeholder(dtype=tf.float32)
placeholder_b = tf.placeholder(dtype=tf.float32)

run(placeholder_a * placeholder_b, {placeholder_a: 3, placeholder_b: 4.5})

# ====================
# パラメータ (変数)
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

liner_calc = W * x + b # y = W * x + b
init = tf.global_variables_initializer() # 変数 (W, b) の初期化
sess.run(init)

run(liner_calc, {x: [1,2,3,4]}) # y = W * x + b にて、 x = 1, 2... の答えを出力

# ====================
# モデルの評価
y = tf.placeholder(tf.float32)
deltas = tf.square(liner_calc - y) # 二乗誤差
loss = tf.reduce_sum(deltas) # delta の合計値計算

# 実際の答えは [0, 0.3, 0.6, 0.9] なので、誤差が大きい = loss が大きい
run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]})

# W, b に対して、再割り当て
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

# 答えと一致するので loss = 0
run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]})

# loss を自動で最小化する
fixW2 = tf.assign(W, [.3])
fixb2 = tf.assign(b, [-.3])
sess.run([fixW2, fixb2])
run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]})

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]})

# 結果
run([W, b])

