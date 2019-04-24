import numpy as np
import matplotlib.pyplot as plt

def padding(m, num):
    shape=(m.shape[0]+num*2, m.shape[1]+num*2)
    b = np.zeros(shape)
    b[num:m.shape[0] + num, num:m.shape[1] + num] = m
    return b


"""
    Generator to extract sub matrices of m, with size subsize x subsize. Moving stridesize, each time.
"""
def extractSubMatrix(m, subSize, strideSize):
    # sub size integer
    dimRow = m.shape[0]
    dimCol = m.shape[1]
    row, col, rowIm, colIm = 0,0,0,0
    while row <= dimRow - subSize:
        col = 0
        colIm = 0
        while col <= dimCol - subSize:
            yield m[row:row+subSize, col:col+subSize], rowIm, colIm
            col+=strideSize
            colIm+=1
        row+=strideSize
        rowIm +=1


"""
 Calculate the dimension of the output matrices in a convultional layer. The number of this matrixes will equal the 
 number of kernels/filters.
"""
def outputDim(input_size, kernel_window, padding, stride):
    res = (input_size - kernel_window + 2*padding)/stride + 1
    return int(res)


def getMaxMask(m, subSize):
    dimRow = m.shape[0]
    dimCol = m.shape[1]
    maxCords = []
    row, col = 0, 0
    while row < dimRow:
        col = 0
        while col < dimCol:
            # Find maximum value in sub-matrix
            maxVal = m[row, col]
            cord = (row, col)
            for i in range(row, row+subSize):
                for j in range(col, col+subSize):
                    if m[i, j] > maxVal:
                        maxVal = m[i, j]
                        cord = (i, j)
            maxCords.append(cord)
            col += subSize
        row += subSize

    res = np.zeros([dimRow, dimCol])
    for c in maxCords:
        res[c[0], c[1]] = 1
    return res

def reshapeByReplicate(m, newDim):
    multiplier = newDim/m.shape[0]
    return m.repeat(multiplier, axis=0).repeat(multiplier, axis=1)

def vis_square(data, path):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    fig, ax = plt.subplots()
    ax.imshow(data);
    plt.axis('off')
    plt.savefig('{}.png'.format(path))

