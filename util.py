import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

def plot_generic(scores_mean, normal_data, add_to_plot=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    score_min, score_max = np.min(scores_mean), np.max(scores_mean)
    plt.xlim([score_min, score_max])
    plt.ylim([0, 1])
    plt.xlabel("error", fontsize=30)
    plt.ylabel("prob", fontsize=30)
    
    ax.plot(scores_mean, normal_data, 'ko', markersize=4, label="errors")
    if add_to_plot is not None:
        add_to_plot(ax)

    plt.legend(loc=4, fontsize=15, frameon=False)
    plt.show()
    
def plot_uncertainty_bands(scores):
    scores_mean = np.mean(scores, axis=1)
    scores_std = np.std(scores, axis=1)
    scores_mean, scores_std = zip(*sorted(zip(scores_mean, scores_std), reverse=False))
    scores_mean = np.array(scores_mean)
    distr_exp = expon(loc=0, scale=2.7)
    normal_data = distr_exp.pdf(scores_mean)
    scores_std = np.array(scores_std)

    def add_uncertainty(ax):
        ax.plot(scores_mean, normal_data, 'k-', linewidth=2, color="#408765", label="predictive error mean")
        ax.fill_between(scores_mean, normal_data - 2 * scores_std, normal_data + 2 * scores_std, alpha=0.6, color='#86cfac', zorder=5)

    plot_generic(scores_mean, normal_data, add_uncertainty)
    
def inversion_number(u, eta):
    u_r = list(filter(lambda e: e <= eta, u))
    u_i = list(filter(lambda e: e > eta, u))
    n_r = len(u_r)
    n_i = len(u_i)
    inr = 0
    ini = 0
    for i in range(n_r):
        for j in range(i + 1, n_r):
            if (u_r[i] > u_r[j]):
                inr += 1
    
    for i in range(n_i):
        for j in range(i + 1, n_i):
            if (u_i[i] < u_i[j]):
                ini += 1
    if n_r >= 2:
        inr = 2*inr/(n_r*(n_r-1))
    if n_i >= 2:
        ini = 2*ini/(n_i*(n_i-1))
    return inr, ini, (inr + ini)/2