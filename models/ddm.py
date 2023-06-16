import numpy as np
from scipy.stats import norm, expon

class DDM:
    def __init__(self, min_e, t):
        '''

        Parameters
        ----------
        min_e : float
            minimum reconstruction error during training.
        t : float
            optimal threshold for anomaly detection.

        Returns
        -------
        None.

        '''
        
        scale = (t - min_e) / 4
        
        # uncertain
        mu_u = t
        scale_u =  scale
        # normal
        mu_n = min_e
        scale_n = scale
        # abnormal
        mu_a = t + scale
        scale_a = scale
        
        self.dist_n = expon(loc=mu_n, scale=scale_n)
        self.dist_u = norm(loc=mu_u, scale = scale_u)
        self.dist_a = expon(loc=mu_a, scale=scale_a)
    
    def prediction(self, e):
        '''
    
        Parameters
        ----------
        e : float
            reconstruction error of a given input.

        Returns
        -------
        pred : an array of 3 floats summing up to 1
            contaning memberships for normality, uncertainty and anomaly.
        float
            uncertainty value.

        '''
        pred = np.array([self.dist_n.pdf(e), self.dist_u.pdf(e), self.dist_a.cdf(e)])
        pred = pred/np.sum(pred)
        return pred, pred[1]*e
    
     
    
        
        
        
        
        