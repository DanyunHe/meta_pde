import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """


    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    #print(current_dict.keys(), output_dict.keys())
    return output_dict

class LinearLayer(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, input_shape).double())
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(hidden_size).double())

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
        else:
            weight, bias = self.weight, self.bias

        out = F.linear(x, weight, bias)

        return out

class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1dLayer, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size).double())
        nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_channels).double())

    def forward(self, x, params = None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
        else:
            weight, bias = self.weight, self.bias
        
        out = F.conv1d(x, weight, bias)
        return out



class Conv1d_Model(nn.Module):
    def __init__(self):
        super(Conv1d_Model,self).__init__()
        self.num_conv_layers = 3
        self.in_channels = 1
        self.num_filters = 32
        self.kernel_size = 8
        self.output_size = 480
        self.layer_dict = nn.ModuleDict()
        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)
        
    def build_network(self):
        x = torch.zeros(1,1,32).double()
        out = x
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['conv{}'.format(0)] = Conv1dLayer(self.in_channels, self.num_filters, self.kernel_size)
        out = self.layer_dict['conv{}'.format(0)](out)
        for i in range(1,self.num_conv_layers):
            self.layer_dict['conv{}'.format(i)] = Conv1dLayer(self.num_filters, self.num_filters, self.kernel_size)

            out = self.layer_dict['conv{}'.format(i)](out)
        out = torch.flatten(out,1)
        self.layer_dict['linear'] = LinearLayer(out.shape[1],self.output_size)
        out = self.layer_dict['linear'](out)

    def forward(self, x, params = None):
        param_dict = dict()
        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict = params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x
        for i in range(self.num_conv_layers):
            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)])
        out = torch.flatten(out,1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def zero_grad(self, params = None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


