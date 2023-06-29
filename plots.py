import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def heatmap(metrics, filename):


    uq_methods = ["EDL", "MCD", "VAEs", "CP",
                  "EDL+", "MCD+", "VAEs+", "CP+"]
    
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
    
def redm(normal, abnormal, filename, scale_n = 0.002, scale_a = 0.0002):
    E_normal, S_normal, S_n, y_normal = normal
    E_abnormal, S_abnormal, S_a, y_abnormal = abnormal
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(E_normal, y_normal, '-b', label='regularity')
    #axs[0, 0].set_ylim(0, 5)
    axs[0, 0].fill_between(E_normal, y_normal - scale_n * S_normal, y_normal + scale_n * S_normal, alpha=0.6, color='#86cfac', zorder=5)
    axs[0, 1].plot(E_normal, y_normal, '-b', label='regularity')
    #axs[0, 1].set_ylim(0, 5)
    axs[0, 1].fill_between(E_normal, y_normal - scale_n * S_n, y_normal + scale_n * S_n, alpha=0.6, color='#86cfac', zorder=5)
    axs[1, 0].plot(E_abnormal, y_abnormal, '-k', label='regularity')
    #axs[1, 0].set_ylim(-1, 3)
    axs[1, 0].fill_between(E_abnormal, y_abnormal - scale_a * S_abnormal, y_abnormal + scale_a * S_abnormal, alpha=0.6, color='#ffcccc', zorder=5)
    axs[1, 1].plot(E_abnormal, y_abnormal, '-k', label='regularity')
    #axs[1, 1].set_ylim(-1, 3)
    axs[1, 1].fill_between(E_abnormal, y_abnormal - scale_a * S_a, y_abnormal + scale_a * S_a, alpha=0.6, color='#ffcccc', zorder=5)
    plt.savefig(filename+".png", dpi=300 )
    
def training_loss(cp, edl, mcd, vae, dbname="kdd"):
     
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, len(cp)+1)
     
    # Plot and label the training and validation loss values
    plt.plot(epochs, cp, label='CP', color='black')
    plt.plot(epochs, edl, label='EDL', color='red')
    plt.plot(epochs, mcd, label='MCD', color='blue')
    plt.plot(epochs, vae, label='VAEs', color='green')

     
    # Add in a title and axes labels
    plt.title('Training Loss')
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