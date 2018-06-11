import numpy as np

def input_evaluation_set():
    features = {"SepalLenght": np.array([6.4, 5.0]),
                "SepalWidth": np.array([2.8, 2.3]),
                "PetalLength": np.array([5.6, 3.3]),
                "PetalWIdth": np.array([2.2, 1.0])
               }
    labels = np.array([2, 1])
    return features, labels

input_evaluation_set()
