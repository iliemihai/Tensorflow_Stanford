from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import edward as ed
import pandas as pd
import tensorflow as tf
from process_data import process_data
from edward.models import Categorical, Normal
from tensorflow.contrib.distributions import Normal, Bernoulli

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 20000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss
CONTEXT_SIZE = 4
SIGMA = 10.0
NEG_SAMPLES = 20

def bayesemb(batch_gen):

    #step 0 define masks
    p_mask = tf.cast(tf.range(CONTEXT_SIZE/2, BATCH_SIZE + CONTEXT_SIZE/2),tf.int32)
    rows = tf.cast(tf.tile(tf.expand_dims(tf.range(0,CONTEXT_SIZE/2),[0]),[BATCH_SIZE,1]),tf.int32)
    columns = tf.cast(tf.tile(tf.expand_dims(tf.range(0,BATCH_SIZE),[1]),[1,CONTEXT_SIZE/2]),tf.int32)
    ctx_mask = tf.concat([rows+colmns,rows+columns+CONTEXT_SIZE/2+1],1)


    # Step 1: define the placeholders for input and output
    center_words = tf.placeholder(shape=[BATCH_SIZE],dtype=tf.int32,name="X")# size [BATCH_SIZE]
    target_words = tf.placeholder(shape=[BATCH_SIZE,1],dtype=tf.int32,name="Y")# size [BATCH_SIZE,1]

    embedded_vectors = tf.Variable(tf.random_normal([])/, name="embedded_vectors") ###################################################
    context_vectors = tf.Variable(tf.random_normal([])/, name="context_vectors")  #####################################################
  
    #define a Normal distribution as prior
    # this will serve as word embedding
    prior = Normal(loc=0.0,scale=tf.ones([SIGMA]))
    log_prior = tf.reduce_sum(prior.log_prob(embedding_vectors) + prior.log_prob(context_vectors))

    # Taget and Context Indices
    p_idx = tf.gather(,p_mask)####################################################################
    p_emb = tf.squeeze(tf.gather(embedded_vectors, p_idx))
    
    #negative samples
    unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(unigram)),[0]),[BATCH_SIZE,1]) ##########################################
    n_idx = tf.multinomial(unigram_logits, NEG_SAMPLES)
    n_emb = tf.gather(embedded_vectors, n_idx)

    #context
    ctx_idx = tf.squeeze(tf.gather(, ctx_mask)) #####################################################################
    ctx_context = tf.gather(context_vectors, ctx_idx)

    #nat params
    ctx_sum = tf.reduce_sum(ctx_context, [1])
    p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(embedded_vectors,ctx_sum),-1),1)
    n_eta = tf.reduce_sum(tf.multiply(embedded_vectors, tf.tile(tf.expand_dims(ctx_sum,1),[1,NEG_SAMPLES,1])),-1)

    #conditional likelihood
    y_pos = Bernoulli(logits = p_eta)
    y_neg = Bernoulli(logits = n_eta)

    ll_pos = tf.reduce_sum(y_pos.log_prob(1.0))
    ll_neg = tf.reduce_sum(y_neg.log_prob(0.0))

    log_lokelihood = ll_pos + ll_neg

    scale = 1.0*LEN_DATA/BATCH_SIZE
    loss = -(scale*log_lokelihood + log_prior)

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    train_loss = np.zeros(NUM_TRAIN_STEPS)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for index in range(NUM_TRAIN_STEPS):
            for ep in range(NUM_EPOCHS):
                centers, targets = next(batch_gen)
                sess.run([train],feed_dict={words:centers})##################################################################
            _, train_loss[index] = sess.run(train,loss,feed_dict={words:centers})############################################
        print ("iteration {:d} {:d}, train loss {:0.3f}\n".format(index,NUM_TRAIN_STEPS,train_loss[index])
        

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    bayesemb(batch_gen)

if __name__ == "__main__":
    main()
