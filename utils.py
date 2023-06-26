from models.ae import AE
from models.utils import model_train, vae_train
import torch
from models.vae import VAE
import torch.nn as nn
import numpy as np
from models.utils import estimate_optimal_threshold

directory_model = "checkpoints/"
directory_data = "data/"
directory_output = "outputs/"
kdd = "kdd"
nsl = "nsl"
ids = "ids"
sample_size = 5
criterions = [nn.MSELoss()]*(sample_size + 1) + [nn.BCELoss()]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_val_scores(model, criterion, config, X_val, y_val):
    val_score = [criterion(model(x_in.to(device))[0], x_in.to(device)).item() for x_in in X_val]
    params = estimate_optimal_threshold(val_score, y_val, pos_label=1, nq=100)
    eta = params["Thresh_star"]
    np.savetxt(directory_output + config + "_scores_val_" + model.name + ".csv", val_score)
    np.savetxt(directory_output + config + "_threshold_" + model.name + ".csv", [eta])
    
def save_test_scores(model, criterion, config, X_test, y_test):
    test_score = [criterion(model(x_in.to(device))[0], x_in.to(device)).item() for x_in in X_test]
    eta = np.loadtxt(directory_output + config + "_threshold_" + model.name + ".csv")
    eta = eta[0]
    y_pred = np.array(test_score) > eta
    y_pred = y_pred.astype(int)
    np.savetxt(directory_output + config + "_scores_test_" + model.name + ".csv", test_score)
    np.savetxt(directory_output + config + "_labels_test_" + model.name + ".csv", y_pred)
    

def train(batch_size = 32, lr = 1e-5, w_d = 1e-5, momentum = 0.9, epochs = 5):
    
    X_kdd_train = np.loadtxt(directory_data+"kdd_train.csv")
    XY_kdd_val = np.loadtxt(directory_data+"kdd_val.csv")
    X_nsl_train = np.loadtxt(directory_data+"nsl_train.csv")
    XY_nsl_val = np.loadtxt(directory_data+"directory_data+nsl_val.csv")
    X_ids_train = np.loadtxt(directory_data+"ids_train.csv")
    XY_ids_val = np.loadtxt(directory_data+"ids_val.csv")
    
    configs = {kdd: [X_kdd_train, XY_kdd_val],
              nsl: [X_nsl_train, XY_nsl_val],
              ids: [X_ids_train, XY_ids_val]}
    
    for config in configs:
        X_train, XY_val = configs[config]
        X_val, y_val = XY_val[:, :-1], XY_val[:, -1]
        X_train = torch.from_numpy(X_train)
        X_val = torch.from_numpy(X_val)
        
        for single in range(sample_size):
            model_name = "ae_model_"+config+"_"+str(single)
            ae_model = AE(X_train.shape[1], model_name)
            model_train(ae_model, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
            ae_model.save()
            save_val_scores(ae_model, criterions[single], config, X_val, y_val)
                 
        #dropout
        model_name = "ae_dropout_model_"+config
        ae_dropout_model = AE(X_train.shape[1], model_name, dropout = 0)
        model_train(ae_dropout_model, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
        ae_dropout_model.save()
        save_val_scores(ae_dropout_model, criterions[sample_size], config, X_val, y_val)
    
        # VAE
        model_name = "vae_model_"+config
        vae = VAE(X_train.shape[1], model_name)
        vae_train(vae, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
        vae.save()
        save_val_scores(vae, criterions[-1], config, X_val, y_val)

    
def evaluate():
    
    XY_kdd_test = np.loadtxt(directory_data+"kdd_test.csv")
    XY_nsl_test = np.loadtxt(directory_data+"nsl_test.csv")
    XY_ids_test = np.loadtxt(directory_data+"ids_test.csv")
    
    configs = {kdd: XY_kdd_test,
              nsl: XY_nsl_test,
              ids: XY_ids_test}
    
    for config in configs:
        XY_test = configs[config]
        X_test, y_test = XY_test[:, :-1], XY_test[:, -1]
        X_test = torch.from_numpy(X_test)
        
        for single in range(sample_size):
            model_name = "ae_model_"+config+"_"+str(single)
            ae_model = AE(X_test.shape[1], model_name)
            ae_model.load()
            ae_model.to(device)
            save_test_scores(ae_model, criterions[single], config, X_test, y_test)
            
        #dropout
        for single in range(sample_size):
            model_name = "ae_dropout_model_"+config
            ae_dropout_model = AE(X_test.shape[1], model_name, dropout = 0.2)
            ae_dropout_model.load()
            ae_dropout_model.to(device)
            ae_dropout_model.name = model_name + single
            save_test_scores(ae_dropout_model, criterions[sample_size], config, X_test, y_test)
    
        # VAE
        for single in range(sample_size):
            model_name = "vae_model_"+config
            vae = VAE(X_test.shape[1], model_name)
            vae.load()
            vae.to(device)
            vae.name = model_name + single
            save_test_scores(vae, criterions[-1], config, X_test, y_test)




