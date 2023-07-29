import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

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

def rejection_plot(rejection, filename):
    sns.barplot(data=rejection, x="metrics", y="count", hue="indicator")
    plt.savefig("rejection_"+filename+".png", dpi=300)
    plt.show()
    
def redm(params, filename, scale_n = 0.002, scale_a = 0.0002):
    
    grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.6)
    
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
    plt.subplot(grid[0, 0:]).legend(loc='best')
    y_n_min = min(np.min(p_normal - scale_n * S_normal), np.min(p_normal - scale_n * S_n))
    y_n_max = max(np.max(p_normal + scale_n * S_normal), np.max(p_normal + scale_n * S_n))
    plt.subplot(grid[1, 0]).plot(E_normal, p_normal, '-b', label='regularity')
    plt.subplot(grid[1, 0]).set_ylim(y_n_min, y_n_max)
    plt.subplot(grid[1, 0]).fill_between(E_normal, p_normal - scale_n * S_normal, p_normal + scale_n * S_normal, alpha=0.6, color='#86cfac', zorder=5)
    plt.subplot(grid[1, 0]).axvline(x = params.eta, color = 'red', lw=0.5)
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
    plt.subplot(grid[2, 1]).plot(E_abnormal, p_abnormal, '-k', label='regularity')
    plt.subplot(grid[2, 1]).set_ylim(y_a_min, y_a_max)
    plt.subplot(grid[2, 1]).axvline(x = params.eta, color = 'red', lw=0.5)
    plt.subplot(grid[2, 1]).fill_between(E_abnormal, p_abnormal - scale_a * S_a, p_abnormal + scale_a * S_a, alpha=0.6, color='#ffcccc', zorder=5)
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
