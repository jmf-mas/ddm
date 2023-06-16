import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import collections
from tqdm import tqdm, tqdm_notebook
from torch.autograd import Variable as V

class Loader(torch.utils.data.Dataset):
    def __init__(self):
        super(Loader, self).__init__()
        self.dataset = ''
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        row = row.drop(labels={'label'})
        data = torch.from_numpy(np.array(row)).float()
        return data
    
class Train_Loader(Loader):
    def __init__(self):
        super(Train_Loader, self).__init__()
        self.dataset = pd.read_csv(
                       'data/random_train.csv',
                       index_col=False
                       )
        
def model2_train(model, train_loader, l_r =  1e-2, epochs = 5):
    
    model_hist = collections.namedtuple('Model','epoch loss val_loss')
    model_loss = model_hist(epoch = [], loss = [], val_loss = [])

    loss_fx = nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=l_r)
    
    for epoch in tqdm_notebook(range(epochs),position=0, total = epochs):
        losses=[]
        dl = iter(train_loader)
        for t in range(len(dl)):
            # Forward pass: compute predicted y and loss by passing x to the model.
            xt = next(dl)
            y_pred = model(V(xt))
            
            l = loss_fx(y_pred, V(xt))
            losses.append(l)
            optimizer.zero_grad()
    
            # Backward pass: compute gradient of the loss with respect to model parameters
            l.backward()
    
            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            

    

# training the model
def model_train(model, X_loader, l_r = 1e-2, w_d = 1e-5, n_epochs = 1, batch_size = 32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=l_r, weight_decay=w_d)
    criterion = nn.MSELoss(reduction='mean')
    model.train(True)
    for epoch in range(n_epochs):
        epoch_loss = []
        for step, batch in enumerate(X_loader):
            x_in = batch.type(torch.float32)
            x_in = x_in.to(device)
            x_out = model(x_in)
            loss = criterion(x_out, x_in)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            torch.autograd.set_detect_anomaly(True)
            epoch_loss.append(loss.item())
    
        print("epoch {}: {}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
    
   


def model_eval(model, X_test):
    loss_fn = nn.MSELoss()
    model.eval()
    X_pred = model(X_test)
    loss_val = loss_fn(X_test, X_pred)
    return loss_val
