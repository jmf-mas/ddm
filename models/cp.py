import numpy as np

class CP:
    
    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        
    def quantile(self, scores): 
        n = len(scores) 
        q_val = np.ceil((1 - self.alpha) * (n + 1)) / n
        return np.quantile(scores, q_val, method="higher")