import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon
from scipy import stats
from matplotlib.pylab import pltlab

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
    
def inversion_number(E_normal, S_normal, E_abnormal, S_abnormal, eta):
    
    E_na = np.array(list(E_normal) + list(E_abnormal))
    S_na = np.array(list(S_normal) + list(S_abnormal))
    
    ES = np.concatenate((E_na.reshape(-1, 1), S_na.reshape(1, -1).T), axis=1)
    ES_r = np.array(list(filter(lambda e: e[0] <= eta, ES)))
    ES_i = np.array(list(filter(lambda e: e[0] > eta, ES)))
    n_r = len(ES_r)
    n_i = len(ES_i)
    inr = 0
    ini = 0
    for i in range(n_r):
        for j in range(i + 1, n_r):
            if (ES_r[i, 1] > ES_r[j, 1]):
                inr += 1
    
    for i in range(n_i):
        for j in range(i + 1, n_i):
            if (ES_i[i, 1] < ES_i[j, 1]):
                ini += 1
    if n_r >= 2:
        inr = 2*inr/(n_r*(n_r-1))
    if n_i >= 2:
        ini = 2*ini/(n_i*(n_i-1))
    
    
    return inr, ini, (inr + ini)/2

def get_heatmap(metrics, filename):


    uq_methods = ["CP", "CP+", "EDL", "EDL+",
                  "MCD", "MCD+", "VAEs", "VAEs+"]
    
    metrics = np.round(metrics, 2)
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(metrics)
    ax.set_xticks(np.arange(len(uq_methods)))
    ax.set_yticks(np.arange(len(uq_methods)))
    ax.set_xticklabels(uq_methods)
    ax.set_yticklabels(uq_methods)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    for i in range(len(uq_methods)):
        for j in range(len(uq_methods)):
            text = ax.text(j, i, metrics[i, j], ha="center", va="center", color="w")
    
    fig.tight_layout()
    plt.savefig(filename+".png", dpi=300)
    plt.show()
    
def generate_distribution_plots(normal, abnormal, filename):
    E_normal, S_normal, S_n, y_normal = normal
    E_abnormal, S_abnormal, S_a, y_abnormal = abnormal
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(E_normal, y_normal, '-b', label='regularity')
    axs[0, 0].set_ylim(0, 5)
    axs[0, 0].fill_between(E_normal, y_normal - 2 * S_normal, y_normal + 2 * S_normal, alpha=0.6, color='#86cfac', zorder=5)
    axs[0, 1].plot(E_normal, y_normal, '-b', label='regularity')
    axs[0, 1].set_ylim(0, 5)
    axs[0, 1].fill_between(E_normal, y_normal - 2 * S_n, y_normal + 2 * S_n, alpha=0.6, color='#86cfac', zorder=5)
    axs[1, 0].plot(E_abnormal, y_abnormal, '-k', label='regularity')
    axs[1, 0].set_ylim(-1, 3)
    axs[1, 0].fill_between(E_abnormal, y_abnormal - 2 * S_abnormal, y_abnormal + 2 * S_abnormal, alpha=0.6, color='#ffcccc', zorder=5)
    axs[1, 1].plot(E_abnormal, y_abnormal, '-k', label='regularity')
    axs[1, 1].set_ylim(-1, 3)
    axs[1, 1].fill_between(E_abnormal, y_abnormal - 2 * S_a, y_abnormal + 2 * S_a, alpha=0.6, color='#ffcccc', zorder=5)
    plt.savefig(filename+".png", dpi=300 )
    
def plot_training_loss(cp, edl, mcd, vae, dbname="kdd"):

    filename = "checkpoints/"
    # Load the training and validation loss dictionaries
    cp_train_loss = np.loadtxt(filename+"")
    edl_train_loss = np.loadtxt(filename+"")
    mcd_train_loss = np.loadtxt(filename+"")
    vae_train_loss = np.loadtxt(filename+"")
     
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, len(cp_train_loss)+1)
     
    # Plot and label the training and validation loss values
    pltlab.plot(epochs, cp_train_loss, label='Training Loss ')
    pltlab.plot(epochs, edl_train_loss, label='Training Loss ')
    pltlab.plot(epochs, mcd_train_loss, label='Training Loss ')
    pltlab.plot(epochs, vae_train_loss, label='Training Loss ')

     
    # Add in a title and axes labels
    pltlab.title('Training Loss')
    pltlab.xlabel('Epochs')
    pltlab.ylabel('Loss')
     
    # Set the tick locations
    pltlab.xticks(np.arange(0, len(cp_train_loss)+1, 2))
     
    # Display the plot
    pltlab.legend(loc='best')
    pltlab.savefig(filename+dbname+"_training.png", dpi=300 )
    pltlab.show()
