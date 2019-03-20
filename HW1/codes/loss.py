from __future__ import division
import numpy as np
from math import exp, log


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        if input.shape == target.shape:
            batchSize = input.shape[0]
            sum = 0
            for i in range(batchSize):
                squaredNorm = np.linalg.norm(input[i] - target[i])**2
                sum += squaredNorm
            result = sum/2/batchSize
            return result

    def backward(self, input, target):
        if input.shape == target.shape:
            # Derivative of the euclidean loss layer in function of the input (prevision)
            result = (input-target)/target.shape[1]
            return result

class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        pass
        """
        if input.shape == target.shape:
            result = []
            for n in range(input.shape[0]):
                result.append(self.get_en(target[n], input[n]) / input.shape[1])
            result = np.array(result)
            return result
        """

    """

    def get_en(self, tn, xn):
        result = 0
        for k in range(len(tn)):
            result -= tn[k]*log(self.get_h(xn, k))
        return result

    def get_h(self, x, k):
        exp_x = []
        for val in x:
            exp_x.append(exp(val))

        return exp_x[k]/sum(exp_x)
"""


    def backward(self, input, target):
        '''Your codes here'''
        pass
