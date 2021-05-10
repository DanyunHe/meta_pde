import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d_Model(nn.Module):
    def __init__(self):
        super(Conv1d_Model, self).__init__()
        self.num_conv_layes = 3
        in_channels = 32
        num_filters = 32
        kernel_size = 5
        self.weights2 = [nn.Parameter(torch.empty(32,1, kernel_size).double().cuda()),
        nn.Parameter(torch.empty(32, 32, kernel_size).double().cuda()),
        nn.Parameter(torch.empty(32, 32, kernel_size).double().cuda())]
        [nn.init.xavier_uniform_(weight) for weight in self.weights]
        self.bias = [nn.Parameter(torch.zeros(num_filters).double().cuda()),
         nn.Parameter(torch.zeros(num_filters).double().cuda()), 
         nn.Parameter(torch.zeros(num_filters).double().cuda())]
        self.weights_linear = nn.Parameter(torch.empty(480,640).double().cuda())
        self.bias_linear = nn.Parameter(torch.zeros(480).double().cuda())
        nn.init.xavier_uniform_(self.weights_linear)
    # start with just 1st order derivative?

    def forward(self,x, weights_copy=None, bias_copy=None, weights_linear_copy=None, bias_linear_copy=None):
        if weights_copy is not None and bias_copy is not None:
            for i in range(self.num_conv_layes):
                x = F.conv1d(x, weight = weights_copy[i], bias = bias_copy[i])
                x = F.relu(x)
            x = torch.flatten(x,1)
            out = F.linear(x, weight = weight_linear_copy, bias = bias_linear_copy)
        else:
            for i in range(self.num_conv_layes):
                x = F.conv1d(x, weight = self.weights[i], bias = self.bias[i])
                x = F.relu(x)
            x = torch.flatten(x,1)
            # import pdb
            # pdb.set_trace()
            out = F.linear(x, weight = self.weights_linear, bias = self.bias_linear)

        return out
