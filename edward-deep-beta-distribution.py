import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Normal, Empirical, Beta

# consts
N = 10000  # training sample size
N_test = 1000 # test sample size
X_DIM = 500  # input dim
MINI_BATCH_SIZE = 100  # batch_size
N_ITER = 10000 # number of iteration of MCMC
np.random.seed(seed=1)

# true_alpha =  softmax(x).mean()
# true_beta = |x|.mean()
def generate_samples(n):
    x = np.random.normal(size=[n, X_DIM])
    true_alpha = (1.0 / (1.0 + np.exp(-x))).mean()
    true_beta = np.absolute(x).mean()
    y = np.random.beta(a=true_alpha, b=true_beta, size=[n])
    return x, y

X_data, Y_data = generate_samples(N)
X_test, Y_test = generate_samples(N_test)

# a function for generating unbiassed mini batches
def next_batch(mini_batch_size=MINI_BATCH_SIZE):
    indexes = np.random.randint(N, size=mini_batch_size)
    return X_data[indexes], Y_data[indexes]

# define prior distributions over parameters
W_0 = Normal(loc=tf.zeros([X_DIM, 50]), scale=tf.ones([X_DIM, 50]))
W_1 = Normal(loc=tf.zeros([50, 10]), scale=tf.ones([50, 10]))
W_2 = Normal(loc=tf.zeros([10, 2]), scale=tf.ones([10, 2]))
b_0 = Normal(loc=tf.zeros(50), scale=tf.ones(50))
b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10))
b_2 = Normal(loc=tf.zeros(2), scale=tf.ones(2))

# define empirical distributions for prameter inference
q_W0 = Empirical(params=tf.Variable(tf.random_normal([N_ITER, X_DIM, 50])))
q_W1 = Empirical(params=tf.Variable(tf.random_normal([N_ITER, 50, 10])))
q_W2 = Empirical(params=tf.Variable(tf.random_normal([N_ITER, 10, 2])))
q_b0 = Empirical(params=tf.Variable(tf.random_normal([N_ITER, 50])))
q_b1 = Empirical(params=tf.Variable(tf.random_normal([N_ITER, 10])))
q_b2 = Empirical(params=tf.Variable(tf.random_normal([N_ITER, 2])))


def build_deep_Beta_distribution(x):
    h_1 = tf.tanh(tf.matmul(x, W_0) + b_0)
    h_2 = tf.tanh(tf.matmul(h_1, W_1) + b_1)
    h_3 = tf.nn.softplus(tf.matmul(h_2, W_2) + b_2)
    alpha = h_3[:, 0]
    beta = h_3[:, 1]
    return Beta(concentration1=alpha, concentration0=beta)

# prepare placeholders for data points
x_ph = tf.placeholder(tf.float32, [MINI_BATCH_SIZE, X_DIM])
y_ph = tf.placeholder(tf.float32, [MINI_BATCH_SIZE])

# build our model
y = build_deep_Beta_distribution(x_ph)

# initialization of parameter inference
latent_vars = {W_0: q_W0, W_1: q_W1, W_2: q_W2,
               b_0: q_b0, b_1: q_b1, b_2: q_b2}
SGHMC = ed.SGHMC(latent_vars=latent_vars, data={y: y_ph})
SGHMC.initialize(scale={y: float(N) / MINI_BATCH_SIZE}, step_size=0.0001, n_iter=N_ITER)

init = tf.global_variables_initializer()
init.run()
for _ in range(N_ITER):
    X_batch, Y_batch = next_batch(MINI_BATCH_SIZE)
    info_dict = SGHMC.update(feed_dict={x_ph: X_batch, y_ph: Y_batch})
    SGHMC.print_progress(info_dict)


# criticize our model
x_test_ph = tf.placeholder(tf.float32, [N_test, X_DIM])
y_test = build_deep_Beta_distribution(x_test_ph)
dict_swap = {W_0: q_W0.mean(), W_1: q_W1.mean(), W_2: q_W2.mean(),
             b_0: q_b0.mean(), b_1: q_b1.mean(), b_2: q_b2.mean()}
y_post = ed.copy(y_test, dict_swap=dict_swap)
log_loss = ed.evaluate(metrics='mean_squared_error', data={x_test_ph: X_test, y_post: Y_test})
print('[deep Beta distribution] mean_squared_error on test data: ', log_loss)



from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
linear = SGDRegressor(loss='squared_loss')
linear.fit(X_data, Y_data)
linear_loss = mean_squared_error(Y_test, linear.predict(X_test))
print('[linear regression] mean_squared_error on test data: ', linear_loss)

