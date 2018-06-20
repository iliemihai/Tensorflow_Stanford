import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np

import sys
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "./data/TRAIN.csv"
BATCH_SIZE = 1
N_COLUMNS = 37
TRAIN_STEPS = 250
N_SAMPLES = 5
TEST_STEPS = 1459

def batch_generator(filename):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(filename_queue)

    record_defaults = [[1.0] for _ in range(N_COLUMNS)]

    # read in batch rows of data
    content = tf.decode_csv(value, record_defaults=record_defaults)

    #pack all 37 features into a tensor
    features = tf.stack(content[:N_COLUMNS-1])

    #last column to label
    label = content[-1]

    min_after_dequeue = 20

    capacity = 21

    data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return data_batch, label_batch

#define a linear model            
#define placehodelrs
X = tf.placeholder(shape=[BATCH_SIZE, N_COLUMNS-1], dtype=tf.float32, name="X")
Y = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32, name="Y")

#create weight and bias
W = tf.Variable(tf.random_normal([N_COLUMNS-1, BATCH_SIZE], stddev=0.1), name="weight", dtype=tf.float32)
b = tf.Variable(tf.zeros([BATCH_SIZE]), name="bias", dtype=tf.float32)

#Predict
Y_pred = tf.add(tf.matmul(X,W), b)

#loss
loss = tf.square(Y-Y_pred, name="loss")

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    try:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./graphs/linearReg", sess.graph)
        data_batch, label_batch = batch_generator([DATA_PATH])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(TRAIN_STEPS):
            total_loss = 0.0
            for _ in range(N_SAMPLES):
                features, labels = sess.run([data_batch, label_batch])
                y_pred, _, loss_per_step = sess.run([Y_pred, optimizer, loss], feed_dict={X:features, Y:labels})
                #print("y_pred", y_pred)
                total_loss += loss_per_step
            #print("Epoca {0}: {1}".format(i, total_loss/N_SAMPLES))
    except tf.errors.OutOfRangeError:
        print('Done training -- limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
            
        writer.close()

        
    W, b = sess.run([W, b])

#print("avem", W,b)


TEST_PATH = "./data/TEST.csv"

def test_generator(filename):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(filename_queue)

    record_defaults = [[1.0] for _ in range(N_COLUMNS-1)]

    # read in batch rows of data
    content = tf.decode_csv(value, record_defaults=record_defaults)

    #pack all 37 features into a tensor
    features = tf.stack(content[:N_COLUMNS-1])

    min_after_dequeue = 20

    capacity = 21

    data_batch = tf.train.shuffle_batch([features], batch_size=BATCH_SIZE, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return data_batch

X_test = tf.placeholder(shape=[BATCH_SIZE, N_COLUMNS-1], dtype=tf.float32, name="X_test")







train = pd.read_csv('./data/train.csv')
train = train.select_dtypes(exclude=['object'])
train.drop('Id',axis = 1, inplace = True)
train.fillna(0,inplace=True)

test = pd.read_csv('./data/test.csv')
test = test.select_dtypes(exclude=['object'])
ID = test.Id
test.fillna(0,inplace=True)
test.drop('Id',axis = 1, inplace = True)


from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)

import warnings
warnings.filterwarnings('ignore')

col_train = list(train.columns)
col_train_bis = list(train.columns)

col_train_bis.remove('SalePrice')

mat_train = np.matrix(train)
mat_test  = np.matrix(test)
mat_new = np.matrix(train.drop('SalePrice',axis = 1))
mat_y = np.array(train.SalePrice).reshape((1314,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)


predictions = []

with tf.Session() as sess:
    try:
        sess.run(tf.global_variables_initializer())
        data_batch = test_generator([TEST_PATH])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(TEST_STEPS):
                features = sess.run([data_batch])
                features = features[0]
                price = sess.run([tf.add(tf.matmul(X_test, W), b)], feed_dict={X_test:features})
                price = prepro_y.inverse_transform(price[0])
                print(price[0][0])
    except tf.errors.OutOfRangeError:
        print('Done training -- limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)



