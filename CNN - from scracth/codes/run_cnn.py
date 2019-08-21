from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helper import vis_square

train_data, test_data, train_label, test_label = load_mnist_4d('data')

# todo -> apply dropout to avoid overfitting

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', 1, 4, 3, 1, 0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))             # output shape: N x 4 x 14 x 14

model.add(Conv2D('conv2', 4, 4, 3, 1, 0.01))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))             # output shape: N x 4 x 7 x 7

model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', 196, 10, 0.1))

# loss = SoftmaxCrossEntropyLoss(name='loss')
loss = EuclideanLoss(name='loss')

# Training configuration
config = {
    'learning_rate': 0.01,
    'weight_decay': 0.001,
    'momentum': 0.9,
    'batch_size': 30,
    'max_epoch': 1,
    'disp_freq': 5,
}

everyLoss = []

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    l = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    everyLoss = np.concatenate((l, everyLoss))

    LOG_INFO('Testing @ %d epoch...' % (epoch))
    test_net(model, loss, test_data, test_label, config['batch_size'])


everyLoss = list(everyLoss)

fig, ax = plt.subplots()
ax.set(xlabel='Batches', ylabel='Loss')
ax.plot(np.array((range(0, len(everyLoss))))*config['batch_size'], everyLoss)
ax.grid()
fig.savefig('lossPlot')

reluOutput = model.get1stConvLayerOutput()

for i in range(9):
    vis_square(reluOutput[i], 'relu1_output_' + str(i))



