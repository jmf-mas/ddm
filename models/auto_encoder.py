import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    
    def __init__(self, input_dim, is_dropout):
        
        super().__init__()
        self.lin1 = nn.Linear(input_dim, 20)
        self.lin2 = nn.Linear(20, 10)
        self.lin7 = nn.Linear(10, 20)
        self.lin8 = nn.Linear(20, input_dim)
        self.is_dropout = is_dropout
        
        self.drop2 = nn.Dropout(0.05)
        
        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):
        x = F.tanh(self.lin1(data))
        if (self.is_dropout):
            x = self.drop2(F.tanh(self.lin2(x)))
        x = F.tanh(self.lin7(x))
        x = self.lin8(x)
        return x