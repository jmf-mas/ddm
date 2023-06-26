import numpy as np

directory = "data/"
kdd = "kdd_data.csv"
nsl = "nsl_data.csv"
ids = "ids_data.csv"

def process():
    
    data_kdd = np.loadtxt(directory + kdd)
    data_nsl = np.loadtxt(directory + nsl)
    data_ids = np.loadtxt(directory + ids)
    

    