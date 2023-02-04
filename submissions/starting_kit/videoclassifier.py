import numpy as np


# Method 2: Large Scale Video Classif with CNN -> http://vision.stanford.edu/pdf/karpathy14.pdf
# Method 3 : Two stream CNN -> https://arxiv.org/pdf/1406.2199.pdf
# Method 4 : 3D Convolution Network -> https://arxiv.org/pdf/1412.0767.pdf

class VideoClassifier(object):
    def __init__(self):
        pass

    def fit(self, videos: list, y, pred_time: float):
        classes = ["A", "B", "C", "D", "E", "F", "G", "H"]
        self.n_classes = len(classes)
        pass

    def predict(self, videos: list, pred_time: float):
        # random soft proba
        proba = np.random.rand(len(videos), self.n_classes)
        proba /= proba.sum(axis=1)[:, np.newaxis]

        # random hard proba
        # idx = np.random.choice(range(self.n_classes), len(videos))
        # proba = np.zeros((len(videos), self.n_classes))
        # proba[range(len(videos)), idx] = 1
        return proba
