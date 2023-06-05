from collections import defaultdict
import torch
import torch.nn as nn
from ae import AE
import time
from datetime import timedelta
import numpy as np

def train(train, train_loader, K=5, batch_size = 32, lr = 1e-2, w_d = 1e-5, momentum = 0.9, epochs = 15):
    
    metrics = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [AE() for k in range(K)]
    ens_start = time.time()
    for k in range(K):
        models[k].to(device)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(models[k].parameters(), lr=lr, weight_decay=w_d) 
        
        models[k].train()
        train_losses = []
        latents = []
        constructed = []
        ae_start = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            for bx, (data) in enumerate(train):
                sample = models[k](data.to(device))
                loss = criterion(data.to(device), sample)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if epoch == epochs-1:
                    train_losses.append(loss.item())
                    latent = models[k].enc(data.to(device))
                    latents.append(latent)
                    constructed.append(sample)
            epoch_loss = running_loss/len(train_loader)
            metrics['train_loss'].append(epoch_loss)
            
            print('-----------------------------------------------')
            print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch+1,epochs,epoch_loss))
            
        ae_end = time.time()
        print('-----------------------------------------------')
        print('[System Complete: {}]'.format(timedelta(seconds=ae_end-ae_start)))
    ens_end = time.time()
    print('Training completes in {}'.format(timedelta(seconds=ens_end-ens_start)))
    
def test(models, data, device, eps):
    criterion = nn.MSELoss(reduction='none')
    losses = []
    for x in data:
        values = []
        for model in models:
            xi = torch.tensor(x).to(torch.float32)
            xi_ = model.forward(xi.to(device))
            loss = criterion(xi.to(device), xi_)
            loss = torch.mean(loss, 0)
            values.append(loss)
        losses.append(values)
    return losses
    