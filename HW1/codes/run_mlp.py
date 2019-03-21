from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import matplotlib.pyplot as plt

# Pergunta a pôr, é suposto manter o parametros iguais quando comparamos os resultados entre activation/loss functions, ou mudar tudo ao mm tempo?

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here

# You should explore different model architecture
# good width values found:  512, 2048
model = Network()
model.add(Linear('fc1', 784, 500, 0.01))
model.add(Sigmoid('act1'))
#model.add(Relu('act1'))
model.add(Linear('fc2', 500, 10, 0.01))

# loss = EuclideanLoss(name='loss')
loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.01,
    'momentum': 0.5,
    'batch_size': 100,
    'max_epoch': 20,
    'disp_freq': 50,
}



for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    LOG_INFO('Testing @ %d epoch...' % (epoch))
    loss_mean, acc_mean = test_net(model, loss, test_data, test_label, config['batch_size'])

