import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from layer import BatchNorm2d, Reshape
from utils import LOG_INFO, printResults
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
LOG_INTERVAL = 50
EPOCHS = 5

LR = 0.01
MM = 0.9
WD = 0.001

device = 'cpu'
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
       transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])),
    batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_list = []
    acc_list = []
    loss_ = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)           # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).float().mean()
        acc_list.append(acc.item())
        if batch_idx % LOG_INTERVAL == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(loss_list), np.mean(acc_list))
            LOG_INFO(msg)
            loss_.append(np.mean(loss_list))
            loss_list.clear()
            acc_list.clear()
    return loss_

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)           # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

pipelines = [
    {
        'BL': True,
        'LR': 0.09,
        'MM': 0.1,
    },
    {
        'BL': False,
        'LR': 0.09,
        'MM': 0.1,
    }
]

for conf in pipelines:

    if conf['BL']:
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1, padding=1),  # in_channels, out_channels, kernel_size,
            BatchNorm2d(4),  # num_filters aka out_channels
            nn.ReLU(),
            nn.AvgPool2d(2),  # kernel_size
            nn.Conv2d(4, 4, 3, stride=1, padding=1),  # in_channels, out_channels, kernel_size,
            BatchNorm2d(4),  # num_filters aka out_channels
            nn.ReLU(),
            nn.AvgPool2d(2),
            Reshape((-1, 196)),
            nn.Linear(196, 10)
        ).to(device)


    else:
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1, padding=1),  # in_channels, out_channels, kernel_size,
            nn.ReLU(),
            nn.AvgPool2d(2),  # kernel_size
            nn.Conv2d(4, 4, 3, stride=1, padding=1),  # in_channels, out_channels, kernel_size,
            nn.ReLU(),
            nn.AvgPool2d(2),
            Reshape((-1, 196)),
            nn.Linear(196, 10)
        ).to(device)
    optimizer = optim.SGD(model.parameters(),  lr=conf['LR'], momentum=conf['MM'], weight_decay=WD)


    everyLoss = []
    acc = []
    startTime = time.time()
    for epoch in range(1, EPOCHS + 1):
        l = train(model, device, train_loader, optimizer, epoch)
        everyLoss = np.concatenate((everyLoss, l))
        acc.append(test(model, device, test_loader))

    stopTime = time.time()
    totalTime = stopTime-startTime
    printResults(conf, ['BL', 'LR', 'MM'], acc, totalTime, 'cnn')

    everyLoss = list(everyLoss)
    fig, ax = plt.subplots()
    ax.set(xlabel='Epochs', ylabel='Loss')
    ax.plot(np.array((range(0, len(everyLoss))))*TRAIN_BATCH_SIZE, everyLoss)
    ax.grid()
    fig.savefig('lossPlot_cnn' + 'BL='+str(conf['BL'])+' LR='+ str(conf['LR'])+' MM='+str(conf['MM'])+' .jpg')
