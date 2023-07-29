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

def misclassification_rate(y_true, y_pred, E, delta_plus, delta_minus, eta, filename):
    n_pos = len(E[y_true==1])
    n_neg = len(E[y_true==0])
    n_mis = len(E[y_true!=y_pred])
    n_mis_un = len(E[(y_true!=y_pred) & (eta-delta_minus <=E) & (E<= eta + delta_plus)])
    n_pos_un = len(E[(y_true==1) & (eta-delta_minus <=E)])
    n_neg_un = len(E[(y_true==0) & (E<= eta + delta_plus)])
    return n_pos_un/n_pos, n_neg_un/n_neg, n_mis_un/n_mis

def false_alarms_metrics(S, y_pred, y_i, y_test, q=50, dec = 4):
    n = len(y_test)
    std_threshold = np.percentile(S, q)
    y_pred = np.array(y_pred)
    uncertainty = (S > std_threshold).astype(int)
    w_rej = np.count_nonzero((y_pred[y_i] == y_test[y_i]) & (uncertainty == True))/n
    r_rej = np.count_nonzero((y_pred[y_i] != y_test[y_i]) & (uncertainty == True))/n
    w_acc = np.count_nonzero((y_pred[y_i] != y_test[y_i]) & (uncertainty == False))/n
    r_acc = np.count_nonzero((y_pred[y_i] == y_test[y_i]) & (uncertainty == False))/n
    w_rej = np.round(w_rej, dec)
    w_acc = np.round(w_acc, dec)
    r_rej = np.round(r_rej, dec)
    r_acc = np.round(r_acc, dec)
    return {"WREJECT":w_rej*100, "WACCEPT":w_acc*100, "RREJECT":r_rej*100, "RACCEPT":r_acc*100}
        
def false_alarms(S, S_p, y_pred, y_i, y_test, q=50, dec = 4):
    columns = ["count", "metrics", "indicator"]
    f = false_alarms_metrics(S, y_pred, y_i, y_test, q=q, dec = dec)
    f_addon = false_alarms_metrics(S_p, y_pred, y_i, y_test, q=q, dec = dec)
    rejection_info = []
    rejection_info.extend([[f[key], key, "without addon"] for key in f])
    rejection_info.extend([[f_addon[key], key, "with addon"] for key in f_addon])
    rejection_info = pd.DataFrame(rejection_info, columns=columns)
    return rejection_info

    
    