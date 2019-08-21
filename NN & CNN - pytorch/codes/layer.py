import torch
import torch.nn as nn
import numpy as np

# todo, use higher LR when this is set
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm1d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.firstIteration = True

        # Initialize weights and bias
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):

        if self.training:
            mean = torch.mean(x, dim=0, keepdim=True)
            variance = torch.mean((x-mean)**2, dim=0, keepdim=True)

            # Moving statistics
            if self.firstIteration:
                self.firstIteration = False
                self.running_mean = mean
                self.running_var = variance
            else:
                self.running_mean = self.running_mean*self.momentum + mean*(1-self.momentum)
                self.running_var = self.running_var*self.momentum + variance*(1-self.momentum)


        else:
            mean = self.running_mean
            variance = self.running_var

        xNorm = (x - mean) / torch.sqrt(variance + self.eps)
        W = self.weight.unsqueeze(1)
        b = self.bias.unsqueeze(1)

        result = W * xNorm + b
        return result


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        print("Begin")
        print(self.weight)
        self.firstIteration = True
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            print("Train")
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            variance = torch.mean((x-mean)**2, dim=[0, 2, 3], keepdim=True)

            # Moving statistics
            if self.firstIteration:
                self.firstIteration = False
                self.running_mean = mean
                self.running_var = variance
            else:
                self.running_mean = self.running_mean * self.momentum + mean * (1 - self.momentum)
                self.running_var = self.running_var * self.momentum + variance * (1 - self.momentum)

        else:
            mean = self.running_mean
            variance = self.running_var

        xNorm = (x - mean) / torch.sqrt(variance + self.eps)
        # print(xNorm.shape)
        # print(self.weight.shape)
        # print(self.bias.shape)

        w = self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        b = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        result = w * xNorm + b

        return result

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

# Implemented before i discovered Reshape was already provided
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
