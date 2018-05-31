import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

import utils

DATA_FILE = 'data/birth_life_2010.txt'

# In order to use eager execution, `tfe.enable_eager_execution()` must be
# called at the very beginning of a TensorFlow program.
tfe.enable_eager_execution()

data, n_samples = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

w = tfe.Variable(0.0, name="weight", dtype=tf.float32)
b = tfe.Variable(0.0, name="bias", dtype=tf.float32)

def prediction(x):
    Y_predicted = x*w + b
    return Y_predicted

def squared_loss(y, Y_predicted):
    loss = tf.square(y - Y_predicted, name="loss")
    return loss

def hubber_loss(labels, predictions, delta=1.0):
#    residual = tf.abs(predictions-labels)
#    def f1(): return 0.5*tf.square(residual)
#    def f2(): return delta * residual - 0.5 * tf.square(delta)
#    return tf.cond(residual < delta, f1, f2)
    t = labels - predictions
    # Note that enabling eager execution lets you use Python control flow and
    # specificy dynamic TensorFlow computations. Contrast this implementation
    # to the graph-construction one found in `utils`, which uses `tf.cond`.
    return t ** 2 if tf.abs(t) <= delta else delta * (2 * tf.abs(t) - delta)


def train(loss_fn):
    print('Training; loss function: ' + loss_fn.__name__)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    #define loss function to make differentiation
    def loss_for_example(x, y):
        return loss_fn(y, prediction(x))
    
    # `grad_fn(x_i, y_i)` returns (1) the value of `loss_for_example`
    # evaluated at `x_i`, `y_i` and (2) the gradients of any variables used in
    # calculating it.
    
    grad_fn = tfe.implicit_value_and_gradients(loss_for_example)
    
    start = time.time()
    for epoch in range(100):
        total_loss= 0.0
        for x_i, y_i in tfe.Iterator(dataset):
            loss, gradients = grad_fn(x_i, y_i)
            optimizer.apply_gradients(gradients)
            total_loss += loss
            
        if epoch % 10 == 0:
            print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))
    print('Took: %f seconds' % (time.time() - start))
    print('Eager execution exhibits significant overhead per operation. '
          'As you increase your batch size, the impact of the overhead will '
          'become less noticeable. Eager execution is under active development: '
          'expect performance to increase substantially in the near future!')

def main():
    train(hubber_loss)
    plt.plot(data[:,0], data[:,1], 'bo')
    plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r',label="huber regression")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
        
    
