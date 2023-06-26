import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import pltlab

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

def plot_data_set_distribution(X, y, filename):
    plt.scatter(X[y==0, 0], X[y==0, 1], s=3, c='blue', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], s=3, c='red', alpha=0.5)
    plt.savefig(filename+".png", dpi=300 )
    plt.show()