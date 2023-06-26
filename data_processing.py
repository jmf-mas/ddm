import numpy as np
from sklearn.model_selection import train_test_split

data_directory ="data/"
file_extension = ".csv"
kdd = "kdd_data.csv"
nsl = "nsl_data.csv"
ids = "ids_data.csv"

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
    y_test = [0]*n_n_r + [1]*n_a
    X_test = np.concatenate((X_n_r, X_a), axis=0)
    
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_rate, random_state=42)
    XY_test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
    XY_val = np.concatenate((X_val, y_val.reshape(-1, 1)), axis=1)
    np.savetxt(data_directory + filename+"_train"+file_extension, X_train, delimiter=",")
    np.savetxt(data_directory + filename+"_val"+file_extension, XY_val, delimiter=",")
    np.savetxt(data_directory + filename+"_test"+file_extension, XY_test, delimiter=",")

def process():
    
    data_kdd = np.loadtxt(data_directory + kdd)
    data_nsl = np.loadtxt(data_directory + nsl)
    data_ids = np.loadtxt(data_directory + ids)
    

    