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
    
def save_test_scores(model, criterion, config, X_test, y_test, eta):
    test_score = [criterion(model(x_in.to(device))[0], x_in.to(device)).item() for x_in in X_test]
    #eta = eta[0]
    y_pred = np.array(test_score) > eta
    y_pred = y_pred.astype(int)
    np.savetxt(directory_output + config + "_scores_test_" + model.name + ".csv", test_score)
    np.savetxt(directory_output + config + "_labels_test_" + model.name + ".csv", y_pred)
    

def train(batch_size = 32, lr = 1e-5, w_d = 1e-5, momentum = 0.9, epochs = 5):
    
    X_kdd_train = np.loadtxt(directory_data+"kdd_train.csv", delimiter=',')
    XY_kdd_val = np.loadtxt(directory_data+"kdd_val.csv", delimiter=',')
    X_nsl_train = np.loadtxt(directory_data+"nsl_train.csv", delimiter=',')
    XY_nsl_val = np.loadtxt(directory_data+"nsl_val.csv", delimiter=',')
    X_ids_train = np.loadtxt(directory_data+"ids_train.csv", delimiter=',')
    XY_ids_val = np.loadtxt(directory_data+"ids_val.csv", delimiter=',')
    
    configs = {kdd: [X_kdd_train, XY_kdd_val],
              nsl: [X_nsl_train, XY_nsl_val],
              ids: [X_ids_train, XY_ids_val]}
    
    
    for config in configs:
        print("training on "+config+" data set")
        X_train, XY_val = configs[config]
        X_val, y_val = XY_val[:, :-1], XY_val[:, -1]
        
        X_val = X_val.astype('float32')
        X_train = X_train.astype('float32')
        X_train = torch.from_numpy(X_train)
        X_val = torch.from_numpy(X_val)
        
        print("training for EDL")
        for single in range(sample_size):
            print("training for ae_model_"+str(single))
            model_name = "ae_model_"+config+"_"+str(single)
            ae_model = AE(X_train.shape[1], model_name)
            model_train(ae_model, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
            ae_model.save()
            save_val_scores(ae_model, criterions[single], config, X_val, y_val)
            print("training for ae_model_"+str(single)+" done")
            
                 
        #dropout
        print("training MCD")
        model_name = "ae_dropout_model_"+config
        ae_dropout_model = AE(X_train.shape[1], model_name, dropout = 0)
        model_train(ae_dropout_model, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
        ae_dropout_model.save()
        save_val_scores(ae_dropout_model, criterions[sample_size], config, X_val, y_val)
        print("training MCD done")
    
        # VAE
        print("training VAEs")
        model_name = "vae_model_"+config
        vae = VAE(X_train.shape[1], model_name)
        vae_train(vae, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
        vae.save()
        save_val_scores(vae, criterions[-1], config, X_val, y_val)
        print("training VAEs done")
        print("training on "+config+" data set done")
        print("---------------------------------------------------------------------")

    
def evaluate():
    
    XY_kdd_test = np.loadtxt(directory_data+"kdd_test.csv", delimiter=',')
    XY_nsl_test = np.loadtxt(directory_data+"nsl_test.csv", delimiter=',')
    XY_ids_test = np.loadtxt(directory_data+"ids_test.csv", delimiter=',')
    
    
    n = [i for i in range(len(XY_ids_test))]
    selection = np.random.choice(n, size = 70000, replace=False)
    np.savetxt(directory_output + "_selection_ids.csv", selection)
    configs = {kdd: XY_kdd_test,
             nsl: XY_nsl_test,
              ids: XY_ids_test[selection]}
    
    for config in configs:
        print("evaluating "+config+" data set")
        XY_test = configs[config]
        X_test, y_test = XY_test[:, :-1], XY_test[:, -1]
        X_test = X_test.astype('float32')
        X_test = torch.from_numpy(X_test)
        
        print("evaluation for EDL")
        for single in range(sample_size):
            print("evaluation for ae_model_"+str(single))
            model_name = "ae_model_"+config+"_"+str(single)
            eta = np.loadtxt(directory_output + config + "_threshold_" + model_name + ".csv")
            ae_model = AE(X_test.shape[1], model_name)
            ae_model.load()
            ae_model.to(device)
            save_test_scores(ae_model, criterions[single], config, X_test, y_test, eta)
            print("evaluation for ae_model_"+str(single)+" done")
            
        #dropout
        print("evaluation for MCD")
        for single in range(sample_size):
            print("evaluation for ae_dropout_model_"+str(single))
            model_name = "ae_dropout_model_"+config
            eta = np.loadtxt(directory_output + config + "_threshold_" + model_name + ".csv")
            ae_dropout_model = AE(X_test.shape[1], model_name, dropout = 0.2)
            ae_dropout_model.load()
            ae_dropout_model.to(device)
            ae_dropout_model.name = model_name + str(single)
            save_test_scores(ae_dropout_model, criterions[sample_size], config, X_test, y_test, eta)
            print("evaluation for ae_dropout_model_"+str(single)+ " done")
        print("evaluation for MCD done")
    
        # VAE
        print("evaluation for VAEs")
        for single in range(sample_size):
            print("evaluation for vae_model_"+str(single))
            model_name = "vae_model_"+config
            eta = np.loadtxt(directory_output + config + "_threshold_" + model_name + ".csv")
            vae = VAE(X_test.shape[1], model_name)
            vae.load()
            vae.to(device)
            vae.name = model_name + str(single)
            save_test_scores(vae, criterions[-1], config, X_test, y_test, eta)
            print("evaluation for vae_model_"+str(single)+ " done")
        print("evaluation for VAEs done")
        print("evaluating "+config+" data set done")




