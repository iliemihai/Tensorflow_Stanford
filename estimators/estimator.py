from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int, help="batch_size")
parser.add_argument("--train_steps", default=1000, type=int, help="train steps")

def main(argv):
    args = parser.parse_args(argv[1:])

    #fetch data
    (train_X, train_Y), (test_X, test_Y) = iris_data.load_data()

    #descibe how use input
    my_feature_column = []
    for key in train_X.keys():
        my_feature_column.append(tf.feature_column.numeric_column(key=key))

    #build 2 hidden layer DNN 10,10
    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_column, hidden_units=[10,10], n_classes=3)
    #train model
    classifier.train(input_fn=lambda:iris_data.train_input_fn(train_X, train_Y, args.batch_size), steps=args.train_steps)
    #eval model
    eval_result = classifier.evaluate(input_fn=lambda:iris_data.eval_input_fn(test_X, test_Y, args.batch_size))

    print("Test accuracy: {accuracy:0.3f}\n".format(**eval_result))
    expected = ['Setosa', 'Versicolor', 'Virginica']
    features = {"SepalLength": [6.4, 5.0],
                "SepalWidth": [2.8, 2.3],
                "PetalLength": [5.6, 3.3],
                "PetalWidth": [2.2, 1.0]
               }

    predictions = classifier.predict(input_fn=lambda:iris_data.eval_input_fn(features, labels=None, batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"') 

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],100 * probability, expec))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
