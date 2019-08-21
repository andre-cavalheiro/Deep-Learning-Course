import numpy as np
from helper import *
from scipy.signal import convolve2d

def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''

    # input 100x1x28x28
    # W 4x1x3x3
    # b 4,
    # kernel_size 3
    # pad 1
    numInputs = input.shape[0]
    numKernels = W.shape[0]
    strideSize = 1                  # could make it a program parameter

    inputDim_h = input.shape[2]
    inputDim_w = input.shape[3]

    outputDim_h = outputDim(inputDim_h, kernel_size, pad, strideSize)  # 26
    outputDim_w = outputDim(inputDim_w, kernel_size, pad, strideSize)  # 26

    output = np.zeros([numInputs, numKernels, outputDim_h, outputDim_w])

    for inputIt, inputSample in enumerate(input):                  # For each training sample in the batch
        for filterIt in range(numKernels):                         # For each filter
            for chanelIt, m in enumerate(inputSample):             # For each Channel

                chanelSpecificFilter_m = np.rot90(W[filterIt, chanelIt],2)
                m_pad = padding(m, pad)

                output[inputIt, filterIt, :, :] += convolve2d(m_pad, chanelSpecificFilter_m, mode='valid')
        output[inputIt, filterIt, :, :] += b[filterIt]

    return output

def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    # input 100x4x14x14
    # grad output 100x4x14x14
    # W 4x4x3x3
    # b 4,
    # kernel_size 3
    # pad 1

    numInputs = input.shape[0]
    numChanels = input.shape[1]     # 4
    numKernels = W.shape[0]         # 4
    inputDim_h = input.shape[2]
    inputDim_w = input.shape[3]
    strideSize = 1                  # could make it a program parameter

    grad_input = np.zeros([numInputs, numChanels, inputDim_h, inputDim_w])
    grad_W = np.zeros([numKernels, numChanels, kernel_size, kernel_size])
    grad_b = np.zeros(numKernels)

    """
    print("conv2d_backward")
    print(grad_output[0, 0, :, :])
    """

    for inputIt in range(numInputs):                                    # For each training sample in the batch
        for filterIt in range(numKernels):                              # For each filter
            gradSpecific = padding(grad_output[inputIt, filterIt], pad)
            for chanelIt in range(numChanels):                          # For each Channel
                chanelSpecificFilter_m = np.rot90(W[filterIt, chanelIt], 2)
                grad_input[inputIt, chanelIt] += convolve2d(gradSpecific, chanelSpecificFilter_m, mode='valid') # todo fillvalue = 0 ?

    for filterIt in range(numKernels):          # For each filter
        for chanelIt in range(numChanels):      # For each Channel
            for inputIt in range(numInputs):    # For each training sample in the batch
                grad_W[filterIt, chanelIt] += convolve2d(input[filterIt, chanelIt, :, :],  np.rot90(grad_output[filterIt, chanelIt, :, :], 2), mode='valid')

    for filterIt in range(numKernels):
        grad_b[filterIt] = np.sum(grad_output[:, filterIt, :, :])

    grad_input[filterIt, chanelIt] /= numInputs
    grad_W /= numInputs
    grad_b /= numInputs

    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    # input 100x4x14x14
    # kernel_size 2
    # pad 0
    numInputs = input.shape[0]
    numChanels = input.shape[1]
    inputDim_h = input.shape[2]
    inputDim_w = input.shape[3]
    strideSize = kernel_size    # Just for the particular case of pooling

    outputDim_h = outputDim(inputDim_h, kernel_size, pad, strideSize)    # 26
    outputDim_w = outputDim(inputDim_w, kernel_size, pad, strideSize)    # 26

    output = np.zeros([numInputs, numChanels, outputDim_h, outputDim_w])

    for inputIt, inputSample in enumerate(input):       # For each sample
        for chanelIt, m in enumerate(inputSample):      # For each Channel
            import skimage.measure
            output[inputIt, chanelIt] = skimage.measure.block_reduce(input[inputIt, chanelIt], (kernel_size, kernel_size), np.average)
    return output

def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width), exact same as on forward
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    # input 100x4x14x14
    # grad output 100x4x7x7
    # kernel_size 2
    # pad 0

    grad_input = np.kron(grad_output, np.ones((kernel_size, kernel_size)))

    return grad_input

