import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys


import tensorflow as tf

DATA_PATH = "data/train.csv"
BATCH_SIZE = 2
N_FEATURES = 80
TRAIN_STEPS = 1

def batch_generator(filename):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(filename_queue)

    record_defaults = [[1.0] for _ in range(N_FEATURES)]
    record_defaults[2] = [""]
    record_defaults[5] = [""]
    record_defaults[6] = [""]
    record_defaults[7] = [""]
    record_defaults[8] = [""]
    record_defaults[9] = [""]
    record_defaults[10] = [""]
    record_defaults[11] = [""]
    record_defaults[12] = [""]
    record_defaults[13] = [""]
    record_defaults[14] = [""]
    record_defaults[15] = [""]
    record_defaults[16] = [""]
    record_defaults[21] = [""]
    record_defaults[22] = [""]
    record_defaults[23] = [""]
    record_defaults[24] = [""]
    record_defaults[25] = [""]
    record_defaults[27] = [""]
    record_defaults[28] = [""]
    record_defaults[29] = [""]
    record_defaults[30] = [""]
    record_defaults[31] = [""]
    record_defaults[32] = [""]
    record_defaults[33] = [""]
    record_defaults[35] = [""]
    record_defaults[39] = [""]
    record_defaults[40] = [""]
    record_defaults[41] = [""]
    record_defaults[42] = [""]
    record_defaults[53] = [""]
    record_defaults[55] = [""]
    record_defaults[57] = [""]
    record_defaults[58] = [""]
    record_defaults[60] = [""]
    record_defaults[63] = [""]
    record_defaults[64] = [""]
    record_defaults[65] = [""]
    record_defaults[72] = [""]
    record_defaults[73] = [""]
    record_defaults[74] = [""]
    record_defaults[78] = [""]
    record_defaults[79] = [""]

    record_defaults.append([1])

    # read in 10 rows of data
    content = tf.decode_csv(value, record_defaults=record_defaults)

    #convert 5th row to binary
    content[2] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[5] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[6] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[7] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[8] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[9] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[10] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[11] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[12] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[13] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[14] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[15] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[16] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[21] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[22] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[23] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[24] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[25] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[26] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[27] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[28] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[29] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[30] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[31] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[32] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[33] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[35] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[39] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[40] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[41] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[42] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[53] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[55] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[57] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[58] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[60] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[63] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[64] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[65] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[72] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[73] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[74] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[78] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))
    content[79] = tf.cond(tf.equal(content[4], tf.constant("Pave")), lambda: tf.constant(1.0), lambda: tf.constant(0.0))

    #pack all 9 features into a tensor
    features = tf.stack(content[:N_FEATURES])

    #last column to label
    label = content[-1]

    min_after_dequeue = 10*BATCH_SIZE

    capacity = 20 * BATCH_SIZE

    data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch

def generate_batches(data_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(TRAIN_STEPS):
            features, labels = sess.run([data_batch, label_batch])
            print(features," + ", labels)
        coord.request_stop()
        coord.join(threads)

def main():
    data_batch, label_batch = batch_generator([DATA_PATH])
    #generate_batches(data_batch, label_batch)

if __name__ == "__main__":
    main()
