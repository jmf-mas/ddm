import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

def plot_generic(scores_mean, distr_exp, add_to_plot=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    score_min, score_max = np.min(scores_mean), np.max(scores_mean)
    plt.xlim([score_min, score_max])
    plt.ylim([0, 1])
    plt.xlabel("error", fontsize=30)
    plt.ylabel("prob", fontsize=30)
    
    ax.plot(scores_mean, distr_exp, 'ko', markersize=4, label="errors")
    if add_to_plot is not None:
        add_to_plot(ax)

    plt.legend(loc=4, fontsize=15, frameon=False)
    plt.show()
    
def plot_uncertainty_bands(scores):
    scores_mean = np.mean(scores, axis=1)
    scores_std = np.std(scores, axis=1)
    scores_mean, scores_std = zip(*sorted(zip(scores_mean, scores_std), reverse=False))
    scores_mean = np.array(scores_mean)
    distr_exp = expon.pdf(scores_mean)
    scores_std = np.array(scores_std)

    def add_uncertainty(ax):
        ax.plot(scores_mean, distr_exp, 'k-', linewidth=2, color="#408765", label="predictive error mean")
        ax.fill_between(scores_mean, distr_exp - 2 * scores_std, distr_exp + 2 * scores_std, alpha=0.6, color='#86cfac', zorder=5)

    plot_generic(scores_mean, distr_exp, add_uncertainty)