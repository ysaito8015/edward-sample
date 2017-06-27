import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Normal, Empirical
import time

N = 20000  # サンプル数
D = 50  # 特徴量の次元
N_ITER = 10000  # MCMCのiteration
MINI_BATCH_SIZE = 2500  #ミニバッチのサイズ 

# toy dataset. 切片=0はなし.
def build_toy_dataset(N, D, noise_std=0.1):
    w = np.random.randn(D).astype(np.float32)
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.dot(X, w) + np.random.normal(0, noise_std, size=N)
    return w, X, Y

# データ生成。観測値のノイズの分散は既知とする。
w_true, X_data, Y_data = build_toy_dataset(N, D)

# ミニバッチを返す関数 
def next_batch(mini_batch_size=128):
    indexes = np.random.randint(N, size=mini_batch_size)
    return X_data[indexes], Y_data[indexes]

# 観測データを挿入するためのデータを収めるplaceholder
x = tf.placeholder(tf.float32, [MINI_BATCH_SIZE, D])
y_ph = tf.placeholder(tf.float32, [MINI_BATCH_SIZE])

w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
y = Normal(mu=ed.dot(x, w) + b, sigma=tf.ones(MINI_BATCH_SIZE)*0.1)

# 経験分布をposteriorの近似に使う
qw = Empirical(params=tf.Variable(tf.random_normal([N_ITER, D])))
qb = Empirical(params=tf.Variable(tf.random_normal([N_ITER, 1])))

# SGLD法用インスタンス
SGLD = ed.SGLD(latent_vars={w: qw, b: qb}, data={y: y_ph})

# 推論GO 
# data辞書にはobservedな確率変数の観測データを送る。
# xの値は確率変数ではないので、updateの際feed_dictで送る。
SGLD = ed.SGLD(latent_vars={w: qw, b: qb}, data={y: y_ph})
SGLD.initialize(scale={y: float(N) / MINI_BATCH_SIZE}, step_size=0.00001, n_iter=N_ITER)

start = time.time()
init = tf.global_variables_initializer()
init.run()
for _ in tqdm(range(N_ITER)):
    X_batch, Y_batch = next_batch(MINI_BATCH_SIZE)
    _ = SGLD.update(feed_dict={x: X_batch, y_ph: Y_batch})
elapsed_time = time.time() - start
print("elapsed_time:{}".format(elapsed_time))
