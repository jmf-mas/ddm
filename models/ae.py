import torch.nn as nn
import pickle
from pathlib import Path

class AE(nn.Module):
    
    def __init__(self, in_dim, is_dropout, name):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8)
        )
        self.enc_dropout = nn.Sequential(
            nn.Linear(in_dim, 64),
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
            nn.Linear(64, in_dim)
        )
        
        self.dec_dropout  = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 32),
            nn.Dropout(0.5),
            nn.Linear(32, 64),
            nn.Dropout(0.5),
            nn.Linear(64, in_dim)
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
    
    def save(self):
        parent_name = "checkpoints"
        Path(parent_name).mkdir(parents=True, exist_ok=True)
        with open(parent_name+"/"+self.name+".pickle", "wb") as fp:
            pickle.dump(self.state_dict(), fp)

    def load(self):
        with open("checkpoints/"+self.name+".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))
    
    