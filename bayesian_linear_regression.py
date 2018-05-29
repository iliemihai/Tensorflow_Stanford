from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

import edward as ed
from edward.models import Normal

DATA_FILE = "data/fire_theft.xls"

book = xlrd.open_workbook(DATA_FILE,encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])
n_samples = sheet.nrows - 1

#2 create placeholders
x = tf.placeholder(tf.float32, shape=[n_samples,1], name="x")
y_ph = tf.placeholder(tf.float32, shape=[n_samples], name="y")
#3 create weight, bias, initialized to 0
#variables name w and b
w = Normal(loc=tf.zeros(1),scale=tf.ones(1))
b = Normal(loc=tf.zeros(1),scale=tf.ones(1))

#4 predict Y (number of theft) from the number of fire
#variable name Y_predicted
y = Normal(loc=ed.dot(x,w)+b,scale=tf.ones(n_samples))

qw = Normal(loc=tf.Variable(tf.random_normal([1])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

sess = ed.get_session()
tf.global_variables_initializer().run()

a = np.reshape(data.T[0],(42,1))

inference = ed.KLqp({w:qw,b:qb}, data={x:a,y_ph:data.T[1]})
inference.initialize()
inference.run(n_samples=2, n_iter=150)

y_post = Normal(loc=ed.dot(x, qw) + qb, scale=tf.ones(n_samples))
#y_post = ed.copy(y, {w: qw, b: qb})

#plot results
X, Y = data.T[0], data.T[1]
plt.plot(X,Y,"bo",label="Real data")
s1 = 0.0
s2 = 0.0
n_sample = 10
#print(qw.sample(n_samples)[:, 0].eval(), qb.sample(n_samples).eval())

W_, B_ = qw.sample(n_samples)[:, 0].eval(), qb.sample(n_samples).eval()

for x in qw.sample(n_samples)[:, 0].eval():
    s1 += x
for x in qb.sample(n_samples).eval():
    s2 += x

w_samples = s1/n_samples
b_samples = s2/n_samples

print ("samples",w_samples, b_samples)
plt.plot(X, X * W_[0] + B_[0], 'r', label='Predicted data')
#plt.plot(X, X * w_samples + b_samples, 'r', label='Predicted data')
plt.legend()
plt.show()
