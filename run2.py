import pandas as pd
from sklearn.preprocessing import RobustScaler
from models.ae import AE
from models.utils import model_train,  model2_train
from models.auto_encoder import AutoEncoder
import torch
from torch.utils.data import DataLoader, RandomSampler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import collections
import gc
gc.enable()

def scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns =cols)
    return std_df

cat_cols = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'level', 'outcome']
def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = scaling(df_num, num_cols)
    
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1
    
    dataframe = pd.get_dummies(dataframe, columns = ['protocol_type', 'service', 'flag'])
    return dataframe

train_transaction = pd.read_csv('data/transaction/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('data/transaction/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('data/identity/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('data/identity/test_identity.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)

def dropper(column_name, train, test):
    train = train.drop(column_name, axis=1)
    test = test.drop(column_name, axis=1)
    return train, test

del_columns = ['TransactionDT']
for col in del_columns:
    train, test = dropper(col, train, test)

def scaler(scl, column_name, data):
    data[column_name] = scl.fit_transform(data[column_name].values.reshape(-1,1))
    return data

scl_columns = ['TransactionAmt', 'card1', 'card3', 'card5', 'addr1', 'addr2']
for col in scl_columns:
    train = scaler(StandardScaler(), col, train)
    test = scaler(StandardScaler(), col, test)
    
y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity

# Drop target
X_train = train.drop('isFraud', axis=1)
#X_train_fraud = train_fraud.drop('isFraud', axis=1)
X_test = test.copy()

del train, test
    
# TODO: change methods
# Fill in NaNs
X_train = X_train.fillna(-999)
#X_train_fraud = X_train_fraud.fillna(-999)
X_test = X_test.fillna(-999)
train_columns = []
for f in X_train.columns:
    if "_" in f:  
        f = f.replace("_", "")
    train_columns.append(f)
    
X_train.columns = train_columns
test_columns = []
for f in X_train.columns:
    if "_" in f:  
        f = f.replace("_", "")
    elif "-" in f:
        f = f.replace("-", "")
    test_columns.append(f)
X_test.columns = test_columns
# TODO: change to Label Count Endocing
# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values)) #+ list(X_train_fraud[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        #X_train_fraud[f] = lbl.transform(list(X_train_fraud[f].values)) 
        X_test[f] = lbl.transform(list(X_test[f].values)) 
        
gc.collect()

def splitter(data, ratio=0.2):
    num = int(ratio*len(data))
    return data[num:], data[:num]

X_train, X_val = splitter(X_train)
y_train, y_val = splitter(y_train)


xtr = torch.FloatTensor(X_train.values)
xts = torch.FloatTensor(X_test.values)
# X_val: validation data for isFraud == 0
xvl = torch.FloatTensor(X_val.values) 
# X_train_fraud: validation data for isFraud == 1
#xvt = torch.FloatTensor(X_train_fraud.values)

train_loader = DataLoader(xtr,batch_size=1000)
test_loader = DataLoader(xts,batch_size=1000)
val_loader = DataLoader(xvl,batch_size=1000)
#fdl = DataLoader(xvt,batch_size=1000)

print(len(X_train.values), len(X_test.values), len(X_val.values)) #, len(X_train_fraud))
gc.collect()

# Check number of data
print(len(X_train), len(X_val), len(y_train), len(y_val))
model = AutoEncoder(len(X_train.columns), False)


model_hist = collections.namedtuple('Model','epoch loss val_loss')
model_loss = model_hist(epoch = [], loss = [], val_loss = [])


# Utilize a named tuple to keep track of scores at each epoch
model_hist = collections.namedtuple('Model','epoch loss val_loss')
model_loss = model_hist(epoch = [], loss = [], val_loss = [])

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


X_train = X_train.values
X_train = X_train.astype('float32')
X_train = torch.from_numpy(X_train)

batch_size = 32
lr = 1e-3
w_d = 1e-5        
momentum = 0.9   
epochs = 15


ae_model = AE(X_train.shape[1], False)
print(ae_model.state_dict())
model_train(ae_model, X_train, l_r = 1e-2, w_d = 1e-5, n_epochs = 1, batch_size = 32)
print(ae_model.state_dict())


