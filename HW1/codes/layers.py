import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        self._saved_for_backward(input)
        result = np.maximum(0, input)
        assert (result.shape == input.shape)
        return result

    def backward(self, grad_output):
        result = np.array(grad_output, copy=True)  # just converting dz to a correct object.
        result[self._saved_tensor <= 0] = 0
        assert(result.shape == self._saved_tensor.shape)
        return result


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        self._saved_for_backward(input)
        result = 1/(1+np.exp(-input))
        assert (result.shape == input.shape)
        return result

    def backward(self, grad_output):
        s = 1 / (1 + np.exp(-self._saved_tensor))
        result = grad_output*s*(1-s)    # Derivative of sigmoid function w.r.t the input
        assert(result.shape == self._saved_tensor.shape)
        return result

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num

        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        self._saved_for_backward(input)

        result = np.dot(input, self.W) + self.b
        assert (result.shape == (input.shape[0], self.W.shape[1]))

        return result

    def backward(self, grad_output):

        self.grad_W = np.matmul(self._saved_tensor.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=-1, keepdims=True)

        prev_grad_output = np.matmul(grad_output, self.W.T)

        return prev_grad_output

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
   