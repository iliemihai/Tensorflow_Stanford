from __future__ import division, print_function, unicode_literals
import numpy as np
import os

import tensorflow as tf
import numpy as np
import math

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"

def logsum_mog(x, pi, mu1, mu2, sigma1, sigma2):
	return log_sum_exp(tf.log(pi) + log_normal(x, mu1, sigma1),
	                   tf.log(1. - pi) + log_normal(x, mu2, sigma2))

def log_sum_exp(u, v):
	m = tf.maximum(u, v)
	return m + tf.log(tf.exp(u - m) + tf.exp(v - m))

def log_normal(x, mu, sigma):
	return -0.5 * tf.log(2.0 * math.pi) - tf.log(tf.abs(sigma)) - tf.square((x - mu)) / (
		2 * tf.square(sigma))

def compute_KL(shape, mu, sigma, prior, sample):
	"""
	Compute KL divergence between posterior and prior.
	"""
	posterior = tf.contrib.distributions.Normal(mu, sigma)
	KL = tf.reduce_sum(posterior.log_prob(tf.reshape(sample, [-1])))
	N1 = tf.contrib.distributions.Normal(0.0, prior.sigma1)
	N2 = tf.contrib.distributions.Normal(0.0, prior.sigma2)
	mix1 = tf.reduce_sum(N1.log_prob(sample), 1) + tf.log(prior.pi_mixture)
	mix2 = tf.reduce_sum(N2.log_prob(sample), 1) + tf.log(1.0 - prior.pi_mixture)
	prior_mix = tf.stack([mix1, mix2])
	KL += -tf.reduce_sum(tf.reduce_logsumexp(prior_mix, [0]))
	return KL





def get_bbb_variable(shape, name, prior, is_training, rho_min_init, rho_max_init):

	with tf.variable_scope('BBB', reuse=not is_training):
		mu = tf.get_variable(name + '_mean', shape, dtype=tf.float32)

	if rho_min_init is None or rho_max_init is None:
		rho_max_init = math.log(math.exp(prior.sigma_mix / 2.0) - 1.0)
		rho_min_init = math.log(math.exp(prior.sigma_mix / 4.0) - 1.0)

	init = tf.random_uniform_initializer(rho_min_init, rho_max_init)

	with tf.variable_scope('BBB', reuse=not is_training):
		rho = tf.get_variable(
			name + '_rho', shape, dtype=tf.float32, initializer=init)

	if is_training or FLAGS.inference_mode == 'sample':
		epsilon = tf.contrib.distributions.Normal(0.0, 1.0).sample(shape)
		sigma = tf.nn.softplus(rho) + 1e-5
		output = mu + sigma * epsilon
	else:
		output = mu

	if not is_training:
		return output


	sample = output
	kl = compute_KL(shape, tf.reshape(mu, [-1]), tf.reshape(sigma, [-1]), prior, sample)
	tf.add_to_collection('KL_layers', kl)
	return output


class Prior(object):
	def __init__(self, pi, log_sigma1, log_sigma2):
		self.pi_mixture = pi
		self.log_sigma1 = log_sigma1
		self.log_sigma2 = log_sigma2
		self.sigma1 = tf.exp(log_sigma1)
		self.sigma2 = tf.exp(log_sigma2)

		sigma_one, sigma_two = math.exp(log_sigma1), math.exp(log_sigma2)
		self.sigma_mix = np.sqrt(pi * np.square(sigma_one) + (1.0 - pi) * np.square(sigma_two))

	def lstm_init(self):
		"""Returns parameters to use when initializing \theta in the LSTM"""
		rho_max_init = math.log(math.exp(self.sigma_mix / 2.0) - 1.0)
		rho_min_init = math.log(math.exp(self.sigma_mix / 4.0) - 1.0)
		return rho_min_init, rho_max_init


class BayesianLSTM(tf.contrib.rnn.BasicLSTMCell):
	def __init__(self, num_units, prior, is_training, inference_mode, bbb_bias,
	             forget_bias=1.0, state_is_tuple=True, activation=tf.tanh,
	             reuse=None, name=None):
		super(BayesianLSTM, self).__init__(num_units, forget_bias, state_is_tuple, activation,
		                                   reuse=reuse)

		self.prior = prior
		self.bbb_bias = bbb_bias
		self.is_training = is_training
		self.h_dim = num_units
		self.inference_mode = inference_mode
		self.theta = None
		self.b = None
		#self.name = name

	def _output(self, theta, b, inputs, h):
		xh = tf.concat([inputs, h], 1)
		return tf.matmul(xh, theta) + tf.squeeze(b)

	def call(self, inputs, state):
		if self.theta is None:
			# Fetch initialization params from prior
			rho_min_init, rho_max_init = self.prior.lstm_init()

			embed_dim = inputs.get_shape()[-1].value
			self.theta = get_bbb_variable((embed_dim + self.h_dim, 4 * self.h_dim),
			                              name=self.name + '_theta',
			                              prior=self.prior,
			                              is_training=self.is_training,
			                              rho_min_init=rho_min_init,
			                              rho_max_init=rho_max_init)

			if self.bbb_bias:
				self.b = get_bbb_variable((4 * self.h_dim, 1),
				                          name=self.name + '_b',
				                          prior=self.prior,
				                          is_training=self.is_training,
				                          rho_min_init=rho_min_init,
				                          rho_max_init=rho_max_init)
			else:
				self.b = tf.get_variable(self.name + '_b', (4 * self.h_dim, 1), tf.float32,
				                         tf.constant_initializer(0.))

		if self._state_is_tuple:
			c, h = state
		else:
			c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

		concat = self._output(self.theta, self.b, inputs, h)
		i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

		new_c = (
			c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
		new_h = self._activation(new_c) * tf.sigmoid(o)

		if self._state_is_tuple:
			new_state = tf.contrib.rnn.LSTMStateTuple(c=new_c, h=new_h)
		else:
			new_state = tf.concat(values=[new_c, new_h], axis=1)

		return new_h, new_state










def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")


#save_fig("time_series_plot")
plt.show()

X_batch, y_batch = next_batch(1, n_steps)
np.c_[X_batch[0], y_batch[0]]

reset_graph()

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

prior = Prior(0.25, -1.0, -7.0)

#cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),output_size=n_outputs)
cell = tf.contrib.rnn.OutputProjectionWrapper(BayesianLSTM(num_units=200, prior=prior, is_training=True, inference_mode="mu", bbb_bias='store_true', forget_bias=1.0, state_is_tuple=True, activation=tf.tanh, reuse=None, name='bbb_lstm_'), output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_iterations = 300
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    saver.save(sess, "./my_time_series_model") # not shown in the book

with tf.Session() as sess:                          # not shown in the book
    saver.restore(sess, "./my_time_series_model")   # not shown

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})


plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

#save_fig("time_series_pred_plot")
plt.show()
