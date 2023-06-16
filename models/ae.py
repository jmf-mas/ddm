import torch.nn as nn

class AE(nn.Module):
    
    def __init__(self, input_dim, is_dropout):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8)
        )
        self.linear =  nn.Linear(16, 8)
        self.dropout = nn.Dropout(0.2)
        self.is_dropout = is_dropout
        self.dec = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        encode = self.enc(x)
        if self.is_dropout:
            encode = self.dropout(self.linear(encode))
        decode = self.dec(encode)
        return decode