import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d_Model(nn.Module):
    def __init__(self, input_size, weights, bias):
        super(Conv1d_Model, self).__init__()
        self.num_conv_layes = 3
        self.weights = weights
        self.bias = bias
    
    # start with just 1st order derivative?

    def forward(self,x, weights_copy=None, bias_copy=None, weights_linear=None, bias_linear=None):
        if weights_copy is not None and bias_copy is not None:
            for i in range(self.num_conv_layes):
                x = F.conv1d(x, weights = weights_copy[i], bias = bias_copy[i])
                x = F.relu(x)
        else:
            for i in range(self.num_conv_layes):
                x = F.conv1d(x, weights = self.weights[i], bias = self.bias[i])
                x = F.relu(x)

        x = torch.flatten(x,1)
        out = self.fc1(x)
        return out
