import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def hubber_loss(labels, prediction, delta=1.0):
    residual = tf.abs(predictions - labels)
    def f1(): return 0.5*tf.square(residual)
    def f2(): return delta*residual - 0.5*tf.square(residual)
    return tf.cond(residual < delta, f1, f2)

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
    	pass
    