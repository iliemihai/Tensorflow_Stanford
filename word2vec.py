import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

import utils
from process_data import process_data

#hyperparams for model
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128
SKIP_WINDOW = 1 
NUM_SAMPLED = 64
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000

def word2vec(batch_gen):
    #1 define placeholders for input and output
    #center_words have to be int
    with tf.name_scope("data"):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="center_words")
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name="target_words")

    #2 define weigths vocab_size*embed_size.initialize with randomUniform
    with tf.name_scope("embedding_matrix"):
        embed_matrix = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name="embedded_matrix")

    #3 define inference
    # get embed of input by lookup in the embedding matrix
    with tf.name_scope("loss"):
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name="embed")
        #4 construct variables for NCE loss
        #nce weights, bias
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0/(EMBED_SIZE**0.5)), name="nce_weight")
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name="nce_bias")

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed, num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE), name="loss")
        #define loss function to be nce loss
        #need to get mean accross all batch
        #use embeddings for center words for inputs not the words themselves

    #5 define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        #initialize all varaibles
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0
        writer = tf.summary.FileWriter("graphs/emb",sess.graph)
        for i in range(NUM_TRAIN_STEPS):
            center, targets = next(batch_gen)
            #create feed_dict,run optimizer, fetch loss batch
            loss_batch, _ = sess.run([loss, optimizer], feed_dict={center_words: center, target_words: targets})

            total_loss += loss_batch
            if (i + 1)%SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(i, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close() 
    
def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == "__main__":
    main()
