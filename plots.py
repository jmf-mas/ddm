import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy
from metrics import false_alarms

def heatmap(metrics, filename):


    uq_methods = ["EDL", "MCD", "VAEs", "CP",
                  "EDL+", "MCD+", "VAEs+", "CP+"]

 
    # Create a dataset
    metrics = np.round(metrics, 2)
    df = pd.DataFrame(metrics, columns=uq_methods, index = uq_methods)

    # plot using a color palette
    sns.heatmap(df, cmap="YlGnBu", annot=True)
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 90)
    plt.savefig(filename+".png", dpi=300)
    plt.show()
    
    metrics = np.round(metrics, 2)

def rejection_plot(S, S_p, y_edl_pred_0, y_i, y_test, filename, dec = 4):
    
    qs = np.array(range(5, 85, 5))
    qs = qs.reshape((4, 4))
    m,n = qs.shape
    fig, axes = plt.subplots(4, 4, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(hspace=0.40, wspace=0.125)
    for i in range(m):
        for j in range(n):
            rejection = false_alarms(S, S_p, y_edl_pred_0, y_i, y_test, q=qs[i, j], dec = dec)
            sns.barplot(ax=axes[i, j], data=rejection, x="metrics", y="count", hue="indicator")
            axes[i, j].set_title("$\gamma=$"+str(qs[i, j]))
            axes[i, j].legend(loc='best', fontsize=4.5)
            axes[i, j].set(xlabel=None)
            if j!=0:
                axes[i, j].set(ylabel=None)
            if i<3:
                axes[i, j].tick_params(bottom=False)
                axes[i, j].set(xticklabels=[])
            else:
                axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=45)
    plt.savefig("rejection_"+filename+".png", dpi=300)
    plt.show()
    
def redm(params, filename, scale_n = 0.002, scale_a = 0.0002, s = 500000):
    fontsize = 4.5
    grid = plt.GridSpec(5, 2, wspace=0.4, hspace=0.6)
    
    E_normal, S_normal, S_n, p_normal, _, _ = params.normal
    E_abnormal, S_abnormal, S_a, p_abnormal, _, _ = params.abnormal
    
    x = np.concatenate((params.E_minus, params.E_star, params.E_plus))
    x.sort()
    y_n = params.n_model(x)*params.dx_minus
    y_u = params.u_model(x)*params.dx_star
    y_a = params.a_model(x)*params.dx_plus
    y_min, y_max = np.min(y_n), np.max(y_n)
    scaler = MinMaxScaler(feature_range=(y_min, y_max))
    y_u = scaler.fit_transform(y_u.reshape(-1, 1))
    y_a = scaler.fit_transform(y_a.reshape(-1, 1))
    plt.subplot(grid[0, 0:]).plot(x, y_n, color='blue', label ='normality pdf')
    plt.subplot(grid[0, 0:]).plot(x, y_a, color='red', label = 'abnormality pdf')
    plt.subplot(grid[0, 0:]).plot(x, y_u, color='gray', label = 'uncertainty pdf')
    plt.subplot(grid[0, 0:]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[0, 0:]).legend(loc='best', fontsize=fontsize)
    plt.subplot(grid[0, 0:]).set_ylabel("probability", fontsize=fontsize)
    y_n_min = min(np.min(p_normal - scale_n * S_normal), np.min(p_normal - scale_n * S_n))
    y_n_max = max(np.max(p_normal + scale_n * S_normal), np.max(p_normal + scale_n * S_n))
    plt.subplot(grid[1, 0]).plot(E_normal, p_normal, '-b', label='regularity')
    plt.subplot(grid[1, 0]).set_ylim(y_n_min, y_n_max)
    plt.subplot(grid[1, 0]).fill_between(E_normal, p_normal - scale_n * S_normal, p_normal + scale_n * S_normal, alpha=0.6, color='#86cfac', zorder=5)
    plt.subplot(grid[1, 0]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[1, 0]).set_ylabel("normality probability", fontsize=fontsize)
    plt.subplot(grid[1, 1]).plot(E_normal, p_normal, '-b', label='regularity')
    plt.subplot(grid[1, 1]).set_ylim(y_n_min, y_n_max)
    plt.subplot(grid[1, 1]).fill_between(E_normal, p_normal - scale_n * S_n, p_normal + scale_n * S_n, alpha=0.6, color='#86cfac', zorder=5)
    plt.subplot(grid[1, 1]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[2, 0]).plot(E_abnormal, p_abnormal, '-k', label='regularity')
    y_a_min = min(np.min(p_abnormal - scale_a * S_abnormal), np.min(p_abnormal - scale_a * S_a))
    y_a_max = max(np.max(p_abnormal + scale_a * S_abnormal), np.max(p_abnormal + scale_a * S_a))
    plt.subplot(grid[2, 0]).set_ylim(y_a_min, y_a_max)
    plt.subplot(grid[2, 0]).fill_between(E_abnormal, p_abnormal - scale_a * S_abnormal, p_abnormal + scale_a * S_abnormal, alpha=0.6, color='#ffcccc', zorder=5)
    plt.subplot(grid[2, 0]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[2, 0]).set_ylabel("abnormality probability", fontsize=fontsize)
    plt.subplot(grid[2, 1]).plot(E_abnormal, p_abnormal, '-k', label='regularity')
    plt.subplot(grid[2, 1]).set_ylim(y_a_min, y_a_max)
    plt.subplot(grid[2, 1]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[2, 1]).fill_between(E_abnormal, p_abnormal - scale_a * S_a, p_abnormal + scale_a * S_a, alpha=0.6, color='#ffcccc', zorder=5)
    x = list(E_normal)[:]
    x.extend(list(E_abnormal))
    cdf = params.cdf(x)
    plt.subplot(grid[3, 0]).plot(x, cdf, '-k', label='regularity')
    S = list(S_normal)[:]
    S.extend(list(S_abnormal))
    S_p = list(S_n)[:]
    S_p.extend(list(S_a))
    S, S_p = np.array(S), np.array(S_p)
    scale_a *=s
    y_a_min = min(np.min(cdf - scale_a * S), np.min(cdf - scale_a * S_p))
    y_a_max = max(np.max(cdf + scale_a * S), np.max(cdf + scale_a * S_p))
    plt.subplot(grid[3, 0]).set_ylim(y_a_min, y_a_max)
    plt.subplot(grid[3, 0]).fill_between(x, cdf - scale_a * S, cdf + scale_a * S, alpha=0.6, color='#ffcccc', zorder=5)
    plt.subplot(grid[3, 0]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[3, 0]).set_ylabel("abnormality probability", fontsize=fontsize)
    plt.subplot(grid[3, 0]).set_xlabel("reconstruction error", fontsize=fontsize)
    plt.subplot(grid[3, 1]).plot(x, cdf, '-k', label='regularity')
    plt.subplot(grid[3, 1]).set_ylim(y_a_min, y_a_max)
    plt.subplot(grid[3, 1]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[3, 1]).fill_between(x, cdf - scale_a * S_p, cdf + scale_a * S_p, alpha=0.6, color='#ffcccc', zorder=5)
    plt.subplot(grid[3, 1]).set_xlabel("reconstruction error", fontsize=fontsize)
    
    
    p_0 = np.array(cdf)
    p_1 = 1-p_0
    p = np.concatenate((p_0.reshape(-1, 1), p_1.reshape(-1, 1)), axis=1)
    H = entropy(p, base=2, axis=1)
    plt.subplot(grid[4, 0]).plot(x, cdf, '-k', label='regularity')
    plt.subplot(grid[4, 0]).set_ylim(y_a_min, y_a_max)
    plt.subplot(grid[4, 0]).fill_between(x, cdf - scale_a * S, cdf + scale_a * S, alpha=0.6, color='#ffcccc', zorder=5)
    plt.subplot(grid[4, 0]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[4, 0]).set_ylabel("abnormality probability", fontsize=fontsize)
    plt.subplot(grid[4, 0]).set_xlabel("reconstruction error", fontsize=fontsize)
    plt.subplot(grid[4, 1]).plot(x, cdf, '-k', label='regularity')
    plt.subplot(grid[4, 1]).set_ylim(y_a_min, y_a_max)
    plt.subplot(grid[4, 1]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[4, 1]).fill_between(x, cdf - scale_a * S*H, cdf + scale_a * S*H, alpha=0.6, color='#ffcccc', zorder=5)
    plt.subplot(grid[4, 1]).set_xlabel("reconstruction error", fontsize=fontsize)
    
    
    
    plt.savefig(filename+".png", dpi=300 )
    
def training_loss(cp, edl, mcd, vae, dbname="kdd"):
     
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, len(cp)+1)
     
    # Plot and label the training and validation loss values
    plt.plot(epochs, cp, label='CP', color='black')
    plt.plot(epochs, edl, label='EDL', color='red')
    plt.plot(epochs, mcd, label='MCD', color='blue')
    #plt.plot(epochs, vae, label='VAEs', color='green')

     
    # Add in a title and axes labels
    #plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
     
    # Set the tick locations
    plt.xticks(np.arange(0, len(cp)+1, 2))
     
    # Display the plot
    plt.legend(loc='best')
    plt.savefig("outputs/"+dbname+"_training.png", dpi=300 )
    plt.show()

def data_set_distribution(X, y, filename):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    plt.scatter(X[y==0, 0], X[y==0, 1], s=3, c='blue', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], s=3, c='red', alpha=0.5)
    plt.savefig(filename+".png", dpi=300 )
    plt.show()
