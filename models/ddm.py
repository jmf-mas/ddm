import numpy as np
from scipy.stats import norm, expon

class DDM:
    def __init__(self, mu_n, mu_u, mu_a, scale_n, scale_u, scale_a):
        
        self.dist_n = expon(loc=mu_n, scale=scale_n)
        self.dist_u = norm(loc=mu_u, scale = scale_u)
        self.dist_a = expon(loc=mu_a, scale=scale_a)
    
    def prediction(self, e):
        pred = np.array([self.dist_n, self.dist_u, self.dist_a])
        pred = pred/np.sum(pred)
        return pred, pred[1]*e
    
     
    