from __future__ import division
import numpy as np
from math import exp, log


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # Mean of sum of squares of differences between inputs and labels
        assert(input.shape == target.shape)
        batchSize = input.shape[0]
        sum = 0
        for i in range(batchSize):
            squaredNorm = np.linalg.norm(input[i] - target[i])**2
            sum += squaredNorm
        result = sum/2/batchSize
        return result

    def backward(self, input, target):
        # Derivative of the euclidean loss function in order to its input (which is actually the output of the NN)
        assert(input.shape == target.shape)
        batchSize = input.shape[0]
        result = (input-target)/batchSize
        assert(result.shape == input.shape)
        return result


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name


    def forward(self, input, target):
        assert(input.shape == target.shape)
        batchSize = input.shape[0]
        result = []
        for n in range(batchSize):
            exp_x = []
            for val in input[n]:
                exp_x.append(exp(val))
            result.append(self._entropy(target[n], input[n], exp_x))
        return sum(result)/batchSize


    def _entropy(self, tn, xn, exp_xn):
        assert(len(tn) == len(xn))
        outputSize = len(tn)
        result = 0
        sumExp = sum(exp_xn)
        for k in range(outputSize):
            result -= tn[k]*log(exp_xn[k]/sumExp)
        return result

    def backward(self, input, target):
        assert(input.shape == target.shape)

        batchSize = input.shape[0]
        numOutputs = input.shape[1]

        inputAfterSoftMax = np.empty(input.shape)
        result = np.empty(input.shape)

        for n in range(batchSize):
            for k in range(numOutputs):
                inputAfterSoftMax[n, k] = self._softmax(input[n], k)
                result[n, k] = inputAfterSoftMax[n, k] - target[n, k]

        assert(inputAfterSoftMax.shape == input.shape)
        return result/batchSize

    def _softmax(self, x, k):
        exp_x = []
        for val in x:
            exp_x.append(exp(val))
        result = exp_x[k]/sum(exp_x)
        return result
