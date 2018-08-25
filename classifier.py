import numpy as np

class classifier:
    """ Interface to be provided by all classifiers"""
    def __init__(self):
        pass
    def predict(self,x):
        raise NotImplementedError("Should have implemented this")

    def loss_function(self,x):
        raise NotImplementedError("Should have implemented this")

class softmax_classifier(classifier):
    """Softmax classifier"""

    def __init__(self):
        classifier.__init__(self)

    # Todo: Investigate on how to reduce risk of numerical errors in softmax
    def predict(self,x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss_function(self,probs,batch_size,y):
        derivative = probs
        derivative[range(batch_size), y] -= 1
        return derivative