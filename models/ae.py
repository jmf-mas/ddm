import torch.nn as nn
import pickle
from pathlib import Path

class AE(nn.Module):
    
    def __init__(self, in_dim, name, dropout = 0):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Linear(64, 32),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.Dropout(dropout),
            nn.Linear(16, 8)
        )
        self.name = name
        
        self.dec  = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 32),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.Dropout(dropout),
            nn.Linear(64, in_dim)
        )
        
    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode
    
    def save(self):
        parent_name = "checkpoints"
        Path(parent_name).mkdir(parents=True, exist_ok=True)
        with open(parent_name+"/"+self.name+".pickle", "wb") as fp:
            pickle.dump(self.state_dict(), fp)

    def load(self):
        with open("checkpoints/"+self.name+".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))
    
    