import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.ae import AE
from models.utils import model_train, vae_train
import torch
from torch.utils.data import DataLoader, RandomSampler
import pickle
from models.vae import VAE
from util import plot_uncertainty_bands
import torch.nn as nn
import numpy as np


def scaling(df_num, cols):
    std_scaler = MinMaxScaler(feature_range=(0, 1))
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns = cols)
    return std_df

cat_cols = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'level', 'outcome']

def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = scaling(df_num, num_cols)
    dataframe = dataframe.reset_index(drop=True)
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1
    dataframe = pd.get_dummies(dataframe, columns = ['protocol_type', 'service', 'flag'])
    return dataframe

data_train = pd.read_csv("data/KDD/KDDTrain.txt")
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])
data_train.columns = columns
data_train.loc[data_train['outcome'] == "normal", "outcome"] = 'normal'
data_train.loc[data_train['outcome'] != 'normal', "outcome"] = 'attack'

#data_train = data_train.loc[data_train['outcome']=='normal']
scaled_train = preprocess(data_train)

batch_size = 32
lr = 1e-5
w_d = 1e-5        
momentum = 0.9   
epochs = 5

train_data_all = scaled_train.sample(frac = 0.9, random_state=200)
train_data = train_data_all.loc[train_data_all['outcome']==0]
val_data = scaled_train.drop(train_data_all.index)
val_data = [val_data, train_data_all.drop(train_data.index)]
val_data = pd.concat(val_data)

X_train = train_data.drop(['outcome', 'level'] , axis = 1).values
X_val = val_data.drop(['outcome', 'level'] , axis = 1).values

y_train = train_data['outcome'].values
y_reg_train = train_data['level'].values
y_val = val_data['outcome'].values
y_reg_val = val_data['level'].values
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_sampler = RandomSampler(X_train)
X_loader = DataLoader(X_train, sampler=X_sampler, batch_size=batch_size)

X_train = torch.from_numpy(X_train)
X_val = torch.from_numpy(X_val)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# classical ae
ae_model = AE(X_train.shape[1], False, "ae_model_kdd")
#model_train(ae_model, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
#ae_model.save()
ae_model.load()
ae_model.to(device)

#dropout
ae_dropout_model = AE(X_train.shape[1], True, "ae_dropout_model_kdd")
#model_train(ae_dropout_model, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
#ae_dropout_model.save()
ae_dropout_model.load()
ae_dropout_model.to(device)

# VAE
vae = VAE(X_train.shape[1], "vae_model_kdd")
#vae_train(vae, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
#vae.save()
vae.load()
vae.to(device)


sample_size = 5
criterion = nn.BCELoss()
#scores = [[criterion(vae(x_in.to(device))[0], x_in.to(device)).item() for i in range(sample_size)] for x_in in X_train]
#scores = np.array(scores)
#np.savetxt("checkpoints/scores_vae_kdd.txt", scores, delimiter=',')
scores = np.loadtxt("checkpoints/scores_vae_kdd.txt", delimiter=',')
plot_uncertainty_bands(scores)




