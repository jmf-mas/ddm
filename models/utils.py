import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import pickle
from pathlib import Path
from sklearn import metrics as sk_metrics

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
    
def model_eval(model, x):
    loss_fn = nn.MSELoss()
    model.eval()
    x_pred = model(x)
    loss_val = loss_fn(x, x_pred)
    return loss_val

def compute_metrics(test_score, y_test, thresh, pos_label=1):
    """
    This function compute metrics for a given threshold

    Parameters
    ----------
    test_score
    y_test
    thresh
    pos_label

    Returns
    -------

    """
    y_pred = (test_score >= thresh).astype(int)
    y_true = y_test.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, test_score)
    roc = sk_metrics.roc_auc_score(y_true, test_score)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm

def estimate_optimal_threshold(test_score, y_test, pos_label=1, nq=100):
    ratio = 100 * sum(y_test == 0) / len(y_test)
    print(f"Ratio of normal data:{ratio}")
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(test_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
        # Prediction using the threshold value
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(test_score, y_test, thresh, pos_label)

        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        # print(f"qi:{qi:.3f} ==> p:{precision:.3f}  r:{recall:.3f}  f1:{f_score:.3f}")
        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }
 
def save(model):
    parent_name = "checkpoints"
    Path(parent_name).mkdir(parents=True, exist_ok=True)
    with open(parent_name+"/"+model.name+".pickle", "wb") as fp:
        pickle.dump(model.state_dict(), fp)

def load(model):
    with open("checkpoints/"+model.name+".pickle", "rb") as fp:
        model.load_state_dict(pickle.load(fp))
    

