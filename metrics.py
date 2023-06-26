from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from  scipy.stats import wilcoxon, friedmanchisquare
import pingouin as pg
import pandas as pd


def confusion_matrix_metrics(y_true, y_pred):
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1}

def inversion_number(E_normal, S_normal, E_abnormal, S_abnormal, eta):
    
    E_na = np.array(list(E_normal) + list(E_abnormal))
    S_na = np.array(list(S_normal) + list(S_abnormal))
    
    ES = np.concatenate((E_na.reshape(-1, 1), S_na.reshape(1, -1).T), axis=1)
    ES_n = np.array(list(filter(lambda e: e[0] <= eta, ES)))
    ES_a = np.array(list(filter(lambda e: e[0] > eta, ES)))
    n_n = len(ES_n)
    n_a = len(ES_a)
    inn = 0
    ina = 0
    for i in range(n_n):
        for j in range(i + 1, n_n):
            if (ES_n[i, 1] > ES_n[j, 1]):
                inn += 1
    
    for i in range(n_a):
        for j in range(i + 1, n_a):
            if (ES_a[i, 1] < ES_a[j, 1]):
                ina += 1
    if n_n >= 2:
        inn = 2*inn/(n_n*(n_n-1))
    if n_a >= 2:
        ina = 2*ina/(n_a*(n_a-1))
    
    
    return inn, ina, (inn + ina)/2

def wilcoxon_test(S1, S2):
    res = wilcoxon(S1, S2)
    return res.pvalue

def friedman_test_for_4_samples(S1, S2, S3, S4):
    res = friedmanchisquare(S1, S2, S3, S4)
    return res.pvalue

def friedman_test_for_8_samples(S1, S2, S3, S4, S5, S6, S7, S8):
    res = friedmanchisquare(S1, S2, S3, S4, S5, S6, S7, S8)
    return res.pvalue

def effect_size(S):
    """

    Parameters
    ----------
    S : Array like (n, m), n rows and m samples
        DESCRIPTION.

    Returns
    -------
    Float
        DESCRIPTION.

    """
    n, m = S.shape
    df = pd.DataFrame({
        'S_'+str(j): {i: S[i, j] for i in range(n)} for j in range(m)})
    res = pg.friedman(df)
    return res.loc['Friedman', 'W']