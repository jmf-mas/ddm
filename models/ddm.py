import numpy as np
from scipy.stats import norm, expon

class DDM:
    def __init__(self, Es, eta, filename, d=0.3, phi=0.2):

        self.Es = Es
        self.eta = eta
        self.filename = filename
        self.d = d
        self.phi = phi
    
    
    def optimal_params(self, E_minus, E_star, E_plus):
        normal_params = expon.fit(E_minus)
        uncertain_params = norm.fit(E_star)
        abnormal_params = expon.fit(E_plus)
        return normal_params, uncertain_params, abnormal_params
    
    def division(self, X, Y):
        R = []
        for x, y in zip(X, Y):
            xy = x/y
            R.append(list(xy))
        return np.array(R)
    
    def distribution_segments(self):
    
        E = np.mean(self.Es, axis=1)
        S = np.std(self.Es, axis=1)
        
        min_ei = np.min(E)
        max_ei = np.max(E)
        
        d_minus = self.eta - min_ei
        d_plus = max_ei - self.eta
        
        delta_minus = self.d*d_minus
        delta_plus = self.d*d_plus
        
        ES = np.concatenate((E.reshape(-1, 1), S.reshape(1, -1).T), axis=1)
        # for plots
        ES_normal = np.array(list(filter(lambda esi: esi[0] < self.eta, ES)))
        ES_abnormal = np.array(list(filter(lambda esi: esi[0] >= self.eta, ES)))
        E_normal, S_normal = ES_normal[:, 0], ES_normal[:, 1]
        E_abnormal, S_abnormal = ES_abnormal[:, 0], ES_abnormal[:, 1]
        # for getting distribution parameters
        ES_minus = np.array(list(filter(lambda esi: esi[0] < self.eta - (1-self.phi)*delta_minus, ES)))
        ES_plus = np.array(list(filter(lambda esi: esi[0] > self.eta + (1-self.phi)*delta_plus, ES)))
        ES_star = np.array(list(filter(lambda esi: self.eta - (1+self.phi)*delta_minus <= esi[0] <= self.eta + (1+self.phi)*delta_plus, ES)))
        E_minus = ES_minus[:, 0]
        E_plus = ES_plus[:, 0]
        E_star = ES_star[:, 0]
        normal_params, uncertain_params, abnormal_params = self.optimal_params(E_minus, E_star, E_plus)
        
        normal_model = lambda x: expon.pdf(x, loc=normal_params[0], scale=normal_params[1])
        abnormal_model = lambda x: expon.cdf(x, loc=abnormal_params[0], scale=abnormal_params[1])
        uncertain_model = lambda x: norm.pdf(x, loc=uncertain_params[0], scale=uncertain_params[1])
        
        E_normal, S_normal = zip(*sorted(zip(E_normal, S_normal), reverse=False))
        E_abnormal, S_abnormal = zip(*sorted(zip(E_abnormal, S_abnormal), reverse=False))
        
        E_normal = list(E_normal)
        S_normal = np.array(list(S_normal))
        E_abnormal = list(E_abnormal)
        S_abnormal = np.array(list(S_abnormal))
        
        y_n_n, y_a_n, y_u_n = normal_model(E_normal), abnormal_model(E_normal), uncertain_model(E_normal)
        y_n_a, y_a_a, y_u_a = normal_model(E_abnormal), abnormal_model(E_abnormal), uncertain_model(E_abnormal)
        
        y_n = np.concatenate((y_n_n.reshape(-1, 1), y_a_n.reshape(1, -1).T), axis=1)
        y_n = np.concatenate((y_n, y_u_n.reshape(1, -1).T), axis=1)
        sum_n = np.sum(y_n, axis=1)
        y_n = self.division(y_n, sum_n)
        
        y_a = np.concatenate((y_n_a.reshape(-1, 1), y_a_a.reshape(1, -1).T), axis=1)
        y_a = np.concatenate((y_a, y_u_a.reshape(1, -1).T), axis=1)
        sum_a = np.sum(y_a, axis=1)
        y_a = self.division(y_a, sum_a)
        
        S_n = np.multiply(y_n[:, 2], S_normal)
        S_a = np.multiply(y_a[:, 2], S_abnormal)
    
        y_normal = normal_model(E_normal)
        y_abnormal = abnormal_model(E_abnormal)   
        return (E_normal, S_normal, S_n, y_normal), (E_abnormal, S_abnormal, S_a, y_abnormal)
    
     
    
        
        
        
        
        