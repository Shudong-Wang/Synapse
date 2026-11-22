import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, **kwargs):
        super(DNN, self).__init__()

        layers = []
        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(p=0.05))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
    
    # def forward(self, x):
    def forward(self, x, g, v=None, mask=None, uu=None, uu_idx=None):
        return self.network(g)