import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.ae import AE
import torch
from torch.utils.data import DataLoader, RandomSampler
import pickle

def scaling(df_num, cols):
    std_scaler = MinMaxScaler(feature_range=(-1, 1))
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
data_train = data_train.loc[data_train['outcome']=='normal']
scaled_train = preprocess(data_train)

X = scaled_train.drop(['outcome', 'level'] , axis = 1).values
y = scaled_train['outcome'].values
y_reg = scaled_train['level'].values
X = X.astype('float32')
X_sampler = RandomSampler(X)
X_loader = DataLoader(X, sampler=X_sampler, batch_size=64)

X_train = X.astype('float32')
X_train = torch.from_numpy(X_train)

batch_size = 32
lr = 1e-5
w_d = 1e-5        
momentum = 0.9   
epochs = 10


ae_model = AE(X_train.shape[1], False, "ae_model")
#model_train(ae_model, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
with open("ae_model_0.pickle", "rb") as fp:
    ae_model.load_state_dict(pickle.load(fp))



