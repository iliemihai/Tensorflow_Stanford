import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

import utils

DATA_FILE = "data/fire_theft.xls"

#read data
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows-1

#create placeholders (X) (Y)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

#create weight, bias, initialized to 0
w = tf.Variable(0.0, name="weight", dtype=tf.float32)
b = tf.Variable(0.0, name="bias", dtype=tf.float32)

#predict Y from X
Y_predicted = X*w + b

#use square root error as a loss function
loss = tf.square(Y - Y_predicted, name="loss")
#loss = utils.hubber_loss(Y, Y_predicted)

#use gradient descent + learn_rate 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

#train model
with tf.Session() as sess:
    #initialize all variables
    sess.run(tf.global_variables_initializer()) 
    writer = tf.summary.FileWriter("./graphs/linearReg", sess.graph)
    #train model
    for i in range(50):
        total_loss = 0
        for x,y in data:
            #session run optimizer to minimize loss + fetch data
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print("Epoca {0}: {1}".format(i, total_loss/n_samples))
    writer.close()
    
    w, b = sess.run([w, b])
    
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, "bo", label="Real data")
plt.plot(X, X*w+b, "r", label="Predicted data")
plt.legend()
plt.show()
