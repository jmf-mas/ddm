import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_rows', 1000)

cat_cols_nsl = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'level', 'outcome']
cat_cols_kdd = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'outcome']

nsl_columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])

kdd_columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome'])



data_directory ="data/"
file_extension = ".csv"
kdd = "kdd_data.csv"
nsl = "nsl_data.csv"
ids = "ids_data.csv"


def scaling(df_num, cols):
    std_scaler = MinMaxScaler(feature_range=(0, 1))
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns = cols)
    return std_df


def preprocess_nsl_data(dataframe):
    df_num = dataframe.drop(cat_cols_nsl, axis=1)
    num_cols = df_num.columns
    scaled_df = scaling(df_num, num_cols)
    dataframe = dataframe.reset_index(drop=True)
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1
    dataframe = pd.get_dummies(dataframe, columns = ['protocol_type', 'service', 'flag'])
    return dataframe

def preprocess_ids_data(chunk):
    
    chunk.drop(chunk.loc[chunk["Label"] == "Label"].index, inplace=True)
    chunk['Label'] = chunk['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    chunk = chunk.drop(columns=['Timestamp'], axis=1).astype('float64')
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

    X = chunk.drop(columns=['Label'])
    y = chunk['Label'].values

    mean_imputer_X = SimpleImputer(strategy='mean')
    X_imputed = mean_imputer_X.fit_transform(X)
    minmax = MinMaxScaler(feature_range=(0, 1))
    X_scaled = minmax.fit_transform(X_imputed)
    
    XY = pd.DataFrame(data = X_scaled,  
                  columns = X.columns, index=X.index)
    XY['label'] = y
    
    return XY

def preprocess_kdd_data(dataframe):
    df_num = dataframe.drop(cat_cols_kdd, axis=1)
    num_cols = df_num.columns
    scaled_df = scaling(df_num, num_cols)
    dataframe = dataframe.reset_index(drop=True)
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    dataframe.loc[dataframe['outcome'] == 'normal', 'outcome'] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1
    dataframe = pd.get_dummies(dataframe, columns = ['protocol_type', 'service', 'flag'])
    return dataframe

def save_processed_data(XY, filename, train_rate = .65, val_rate = 0.2):
    X, y = XY[:, :-1], XY[:, -1]
    X_n = X[y==0]
    X_a = X[y==1]
    n_n = len(X_n)
    n_a = len(X_a)
    ranges = [i for i in range(n_n)]
    selected = np.random.choice(ranges, int(train_rate*n_n), replace = False)
    X_train = X_n[selected]
    
    remained = list(set(ranges).difference(set(selected)))
    X_n_r = X_n[remained]
    n_n_r = len(X_n_r)
    y_test = np.array([0]*n_n_r + [1]*n_a)
    X_test = np.concatenate((X_n_r, X_a), axis=0)
    
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_rate, random_state=42)
    XY_test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
    XY_val = np.concatenate((X_val, y_val.reshape(-1, 1)), axis=1)
    np.savetxt(data_directory + filename+"_train"+file_extension, X_train, delimiter=",")
    np.savetxt(data_directory + filename+"_val"+file_extension, XY_val, delimiter=",")
    np.savetxt(data_directory + filename+"_test"+file_extension, XY_test, delimiter=",")

def process_raw_data():
    
    # ids
    print("processing ids")
    file_id='1e-5ky0j6SG5D3ODxkHxe73W6tKErsxNZ'
    dwn_url='https://drive.google.com/uc?id=' + file_id
    data_ids = pd.read_csv(dwn_url)
    data_ids = preprocess_ids_data(data_ids)
    np.savetxt("data/ids.csv", data_ids.values, delimiter=',')
    
    # nsl
    print("processing nsl")
    file_id='1qdhbdnv258fdw7H4_WIYG3RvM19pCjAQ'
    dwn_url='https://drive.google.com/uc?id=' + file_id
    data_train = pd.read_csv(dwn_url, header=None)
    file_id='198OOWc7CQ9nF_AUFVbKT1Smsb-c7FNlM'
    dwn_url='https://drive.google.com/uc?id=' + file_id
    data_test = pd.read_csv(dwn_url, header=None)
    data_nsl = np.concatenate((data_train.values, data_test.values), axis=0)
    data_nsl = pd.DataFrame(data = data_nsl,  columns = nsl_columns)
    data_nsl.loc[data_nsl['outcome'] == "normal", "outcome"] = 'normal'
    data_nsl.loc[data_nsl['outcome'] != 'normal', "outcome"] = 'attack'
    data_nsl = preprocess_nsl_data(data_nsl)
    X = data_nsl.drop(['outcome', 'level'] , axis = 1).values
    y = data_nsl['outcome'].values
    
    data_nsl = np.concatenate((X, y.reshape(1, -1).T), axis=1)
    #np.savetxt("data/nsl.csv", data_nsl, delimiter=',')
    
    # kdd cup
    print("processing kdd")
    file_id='1by815Yv3oUjcW0zwRjVrgfkCuEQ-bxlW'
    dwn_url='https://drive.google.com/uc?id=' + file_id
    data_kdd = pd.read_csv(dwn_url, header=None)
    data_kdd.columns = kdd_columns
    data_kdd.loc[data_kdd['outcome'] == 'normal.', 'outcome'] = 'normal'
    data_kdd.loc[data_kdd['outcome'] != 'normal', 'outcome'] = 'attack'
    data_kdd = preprocess_kdd_data(data_kdd)
    data_kdd = np.concatenate((X, y.reshape(1, -1).T), axis=1)
    #np.savetxt("data/kdd.csv", data_kdd, delimiter=',')
    
def split_and_save_data():
    
    XY = np.loadtxt(data_directory+"kdd.csv", delimiter=',')
    #save_processed_data(XY, "kdd", train_rate = .65, val_rate = 0.2)
    XY = np.loadtxt(data_directory+"nsl.csv", delimiter=',')
    #save_processed_data(XY, "nsl", train_rate = .65, val_rate = 0.2)
    XY = np.loadtxt(data_directory+"ids.csv", delimiter=',')
    save_processed_data(XY, "ids", train_rate = .11, val_rate = 0.3)

    

    