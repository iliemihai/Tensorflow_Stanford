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
WEIGHTS_FLD = 'processed/'

DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000

class SkipGramModel:
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name="center_words")
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="target_words")

    def _create_embeddings(self):
        with tf.name_scope("embeddings"):
            self.embed_matrix = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], -1.0, 1.0), name="embedded_matrix")


    def _create_loss(self):
        with tf.name_scope("loss"):
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name="embed")
            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1.0/(self.embed_size**0.5)), name="nce_weight")
            nce_bias = tf.Variable(tf.zeros([self.embed_size]), name="nce_bias")
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=self.target_words, inputs=embed, num_sampled=self.num_sampled, num_classes=self.vocab_size), name="loss")

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("histogram loss", self.loss)
        # because you have several summaries, we should merge them all
        # into one op to make it easier to manage
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_embeddings()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

def train_model(model, batch_gen, num_train_steps, weights_fld):
    saver = tf.train.Saver() # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias

    initial_step = 0
    utils.make_dir('checkpoints')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('improved_graph/lr' + str(LEARNING_RATE), sess.graph)
        initial_step = model.global_step.eval()
        for index in range(initial_step, initial_step + num_train_steps):
            centers, targets = next(batch_gen)
            feed_dict={model.center_words: centers, model.target_words: targets}
            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], 
                                              feed_dict=feed_dict)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'checkpoints/skip-gram', index)
        
        ####################
        # code to visualize the embeddings. uncomment the below to visualize embeddings
        # run "'tensorboard --logdir='processed'" to see the embeddings
        # final_embed_matrix = sess.run(model.embed_matrix)
        
        # # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        # embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        # sess.run(embedding_var.initializer)

        # config = projector.ProjectorConfig()
        # summary_writer = tf.summary.FileWriter('processed')

        # # add embedding to the config file
        # embedding = config.embeddings.add()
        # embedding.tensor_name = embedding_var.name
        
        # # link this tensor to its metadata file, in this case the first 500 words of vocab
        # embedding.metadata_path = 'processed/vocab_1000.tsv'

        # # saves a configuration file that TensorBoard will read during startup.
        # projector.visualize_embeddings(summary_writer, config)
        # saver_embed = tf.train.Saver([embedding_var])
        # saver_embed.save(sess, 'processed/model3.ckpt', 1)

def main():
    model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)

if __name__ == '__main__':
    main()
