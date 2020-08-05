import torch
import torch.nn as nn 
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, hid_dim=40):
        super(Net, self).__init__()
        self.num_layers = 3
        self.layer_dim = [1, hid_dim, hid_dim, 1]
        self.fcs = nn.ModuleList()
        self._stack_layers()

    def _stack_layers(self):
        assert self.num_layers+1 == len(self.layer_dim)
        for i in range(self.num_layers):
            self.fcs.append(nn.Linear(self.layer_dim[i], self.layer_dim[i+1], bias=True))
 
    def forward(self, x, params=None):
        load = False if params is None else True
        for i in range(self.num_layers):
            if load:
                name = "fcs.%d" %(i)
                x = F.linear(x, params[name+".weight"], params[name+".bias"])
            else:
                x = self.fcs[i](x)

            if i != self.num_layers -1:
                x = F.relu(x)
                
        return x