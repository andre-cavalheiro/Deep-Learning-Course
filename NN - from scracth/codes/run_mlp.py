from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

# Pergunta a pôr, é suposto manter o parametros iguais quando comparamos os resultados entre activation/loss functions, ou mudar tudo ao mm tempo?

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here

# You should explore different model architecture

# Configuration

#usedNN = '1 layer'
usedNN = '2 layers'

usedActivation = 'sigmoid'
#usedActivation = 'relu'

usedLoss = 'euclidean'
#usedLoss = 'crossEntropy'

#width1 = 512
width1 = 1024
#width1 = 2048

#width2 = 256
width2 = 1024
#width2 = 2048

init_std_dev = 1/sqrt(784)
init_std_dev_l1 = 1/sqrt(width1)
init_std_dev_l2 = 1/sqrt(width2)


model = Network()

if usedNN == '1 layer':
    model.add(Linear('fc1', 784, width1, init_std_dev))
    if usedActivation == 'sigmoid':
        model.add(Sigmoid('act1'))
    else:
        model.add(Relu('act1'))
    model.add(Linear('fc2', width1, 10, init_std_dev))
if usedNN == '2 layers':
    model.add(Linear('fc1', 784, width1, init_std_dev))
    if usedActivation == 'sigmoid':
        model.add(Sigmoid('act1'))
    else:
        model.add(Relu('act1'))
    model.add(Linear('fc2', width1, width2, init_std_dev))
    if usedActivation == 'sigmoid':
        model.add(Sigmoid('act2'))
    else:
        model.add(Relu('act2'))
    model.add(Linear('fc3', width2, 10, init_std_dev))


if usedLoss == 'euclidean':
    loss = EuclideanLoss(name='loss')
else:
    loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.001,
    'momentum': 0.5,
    'batch_size': 100,
    'max_epoch': 30,
    'disp_freq': 50,
}

draw_graph = True
loss_vals_train = []
loss_vals_test = []

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    loss_vals = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], draw_graph)
    loss_vals_train = list(np.concatenate((loss_vals_train, loss_vals)))

    draw_graph = False

    LOG_INFO('Testing @ %d epoch...' % (epoch))
    loss_mean, acc_mean = test_net(model, loss, test_data, test_label, config['batch_size'])
    loss_vals_test.append(loss_mean)

# Plot graphs

plt.yticks(np.arange(0, 1, 0.1))       # Must change accordingly

fig = plt.figure()
plt.plot(range(1, len(loss_vals_train)+1), loss_vals_train)
plt.xlabel('Visualised Iterations')
plt.ylabel('Loss')
fig.savefig('plots/train_' + usedNN + '_' + usedActivation + '_' + usedLoss + '.png')

fig = plt.figure()
plt.plot(range(1, config['max_epoch']+1), loss_vals_test)
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig.savefig('plots/test_' + usedNN + '_' + usedActivation + '_' + usedLoss + '.png')

