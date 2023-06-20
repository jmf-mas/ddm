import torch
import torch.nn as nn
import pickle
from pathlib import Path

class VAE(nn.Module):
    def __init__(self, in_dim, name):
        super(VAE, self).__init__()
        self.name = name
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Define the latent representation
        self.fc_mu = nn.Linear(16, 8)
        self.fc_logvar = nn.Linear(16, 8)

        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return mu + std*eps
    
    def save(self):
        parent_name = "checkpoints"
        Path(parent_name).mkdir(parents=True, exist_ok=True)
        with open(parent_name+"/"+self.name+".pickle", "wb") as fp:
            pickle.dump(self.state_dict(), fp)

    def load(self):
        with open("checkpoints/"+self.name+".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))