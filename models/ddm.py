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
        
        ES_minus = np.array(list(filter(lambda esi: esi[0] < self.eta - (1-self.phi)*delta_minus, ES)))
        ES_plus = np.array(list(filter(lambda esi: esi[0] > self.eta + (1-self.phi)*delta_plus, ES)))
        ES_star = np.array(list(filter(lambda esi: self.eta - (1+self.phi)*delta_minus <= esi[0] <= self.eta + (1+self.phi)*delta_plus, ES)))
        E_minus, S_minus = ES_minus[:, 0], ES_minus[:, 1]
        E_plus, S_plus = ES_plus[:, 0], ES_plus[:, 1]
        E_star, S_star = ES_star[:, 0], ES_star[:, 1]
        
        normal_params, uncertain_params, abnormal_params = self.optimal_params(E_minus, E_star, E_plus)
        
        normal_model = lambda x: expon.pdf(x, loc=normal_params[0], scale=normal_params[1])
        abnormal_model = lambda x: expon.cdf(x, loc=abnormal_params[0], scale=abnormal_params[1])
        uncertain_model = lambda x: norm.pdf(x, loc=uncertain_params[0], scale=uncertain_params[1])
        
        E_minus, S_minus = zip(*sorted(zip(E_minus, S_minus), reverse=False))
        E_plus, S_plus = zip(*sorted(zip(E_plus, S_plus), reverse=False))
        E_star, S_star = zip(*sorted(zip(E_star, S_star), reverse=False))
        
        E_minus = list(E_minus)
        S_minus = np.array(list(S_minus))
        E_plus = list(E_plus)
        S_plus = np.array(list(S_plus))
        E_star = list(E_star)
        S_star = np.array(list(S_star))
        
        y_n_minus, y_n_star, y_n_plus = normal_model(E_minus), normal_model(E_star), normal_model(E_plus)
        y_a_minus, y_a_star, y_a_plus = abnormal_model(E_minus), abnormal_model(E_star), abnormal_model(E_plus)
        y_u_minus, y_u_star, y_u_plus = uncertain_model(E_minus), uncertain_model(E_star), uncertain_model(E_plus)
        
        
        y_m = np.concatenate((y_n_minus.reshape(-1, 1), y_a_minus.reshape(1, -1).T), axis=1)
        y_m = np.concatenate((y_m, y_u_minus.reshape(1, -1).T), axis=1)
        sum_m = np.sum(y_m, axis=1)
        y_m = self.division(y_m, sum_m)
    
        y_s = np.concatenate((y_n_star.reshape(-1, 1), y_a_star.reshape(1, -1).T), axis=1)
        y_s = np.concatenate((y_s, y_u_star.reshape(1, -1).T), axis=1)
        sum_s = np.sum(y_s, axis=1)
        y_s = self.division(y_s, sum_s)
        
        y_p = np.concatenate((y_n_plus.reshape(-1, 1), y_a_plus.reshape(1, -1).T), axis=1)
        y_p = np.concatenate((y_p, y_u_plus.reshape(1, -1).T), axis=1)
        sum_p = np.sum(y_p, axis=1)
        y_p = self.division(y_p, sum_p)
        
        S_m = np.multiply(y_m[:, 2], S_minus)
        S_s = np.multiply(y_s[:, 2], S_star)
        S_p = np.multiply(y_p[:, 2], S_plus)
    
        y_minus = normal_model(E_minus)
        y_plus = abnormal_model(E_plus)
        y_star = uncertain_model(E_star)
        
        return (E_minus, S_minus, S_m, y_minus), (E_star, S_star, S_s, y_star), (E_plus, S_plus, S_p, y_plus)
    
    
     
    
        
        
        
        
        