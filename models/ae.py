import torch.nn as nn

class AE(nn.Module):
    
    def __init__(self, input_dim, is_dropout, name):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8)
        )
        self.enc_dropout = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.Dropout(0.5),
            nn.Linear(16, 8)
        )
        self.is_dropout = is_dropout
        self.name = name
        self.dec = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.Linear(64, input_dim)
        )
        
        self.dec_dropout  = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 32),
            nn.Dropout(0.5),
            nn.Linear(32, 64),
            nn.Dropout(0.5),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        encode = None
        decode = None
        if  not self.is_dropout:
            encode = self.enc(x)
            decode = self.dec(encode)
        else:
            encode = self.enc_dropout(x)
            decode = self.dec_dropout(encode)
        return decode