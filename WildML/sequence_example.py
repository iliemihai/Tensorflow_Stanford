import tensorflow as tf
import numpy as np
import tempfile

sequences = [[1,2,3],[4,5,6],[7,8,9]]
labels = [[0,1,0],[1,0,0],[1,1]]

def make_example(sequence, labels):
    #object we return
    ex = tf.train.SequenceExample()

    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for tok, lab in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(tok)
        fl_labels.feature.add().int64_list.value.append(lab)

    return ex

with tempfile.NamedTemporaryFile() as fp:
    writer = tf.python_io.TFRecordWriter(fp.name)
    for seq, lab in zip(sequences, labels):
        ex = make_example(seq, lab)
        writer.write(ex.SerializeToString())
    writer.close()
    print("Wrote to {}".format(fp.name))

tf.reset_default_graph()

ex = make_example([1, 2, 3], [0, 1, 0]).SerializeToString()

# Define how to parse the example
context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

# Parse the example (returns a dictionary of tensors)
context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex,
    context_features=context_features,
    sequence_features=sequence_features
)

context = tf.contrib.learn.run_n(context_parsed, n=1, feed_dict=None)
print(context[0])
sequence = tf.contrib.learn.run_n(sequence_parsed, n=1, feed_dict=None)
print(sequence[0])
