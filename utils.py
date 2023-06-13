import numpy as np
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt
import torch
import tqdm
import torch.nn as nn
from tqdm import trange
from torchvision import datasets
from PIL import Image
from torchvision.transforms.functional import rotate
import torch.optim as optim


def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize = 12)
    plt.show()

def entropy(p):
  # return NotImplemented
  return (-p * np.log(p)).sum(axis=1)

# training the model
def model_train(model, X_loader, n_epochs = 1, batch_size = 64):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss(reduction='mean')
    model.zero_grad()
    model.train(True)
    for epoch in range(n_epochs):
        epoch_loss = []
        for step, batch in enumerate(X_loader):
            x_in = batch.type(torch.float32)
    
            x_out = model(x_in)
            loss = criterion(x_out, x_in)
            print(loss.item())
            loss.backward()
            optimizer.step()
            model.zero_grad()
    
            epoch_loss.append(loss.item())
    
        print("epoch {}: {}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
    
   


def model_eval(model, X_test):
    loss_fn = nn.MSELoss(reduction='mean')
    model.eval()
    X_pred = model(X_test)
    loss_val = loss_fn(X_test, X_pred)
    return loss_val


def train(net, train_data):
    x_train, y_train = train_data
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    progress_bar = trange(3000)
    for _ in progress_bar:
        optimizer.zero_grad()
        loss = criterion(y_train, net(x_train))
        progress_bar.set_postfix(loss=f'{loss / x_train.shape[0]:.3f}')
        loss.backward()
        optimizer.step()
    return net

def plot_predictions(x_test, y_preds):
    def add_predictions(ax):
        ax.plot(x_test, y_preds, 'r-', linewidth=3, label='neural net prediction')

    plot_generic(add_predictions)
    
def plot_uncertainty_bands(x_test, y_preds):
    y_preds = np.array(y_preds)
    y_mean = y_preds.mean(axis=0)
    y_std = y_preds.std(axis=0)
    
    def add_uncertainty(ax):
        ax.plot(x_test, y_mean, '-', linewidth=3, color="#408765", label="predictive mean")
        ax.fill_between(x_test.ravel(), y_mean - 2 * y_std, y_mean + 2 * y_std, alpha=0.6, color='#86cfac', zorder=5)

    plot_generic(add_uncertainty)
    
# for ensemble
def plot_multiple_predictions(x_test, y_preds):
    def add_multiple_predictions(ax):
        for idx in range(len(y_preds)):
            ax.plot(x_test, y_preds[idx], '-', linewidth=3)

    plot_generic(add_multiple_predictions)

def get_simple_data_train():
    x = np.linspace(-.2, 0.2, 500)
    x = np.hstack([x, np.linspace(.6, 1, 500)])
    eps = 0.02 * np.random.randn(x.shape[0])
    y = x + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps)) + eps
    x_train = torch.from_numpy(x).float()[:, None]
    y_train = torch.from_numpy(y).float()
    return x_train, y_train


def plot_generic(add_to_plot=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlim([-.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    x_train, y_train = get_simple_data_train()

    x_true = np.linspace(-.5, 1.5, 1000)
    y_true = x_true + 0.3 * np.sin(2 * np.pi * x_true) + 0.3 * np.sin(4 * np.pi * x_true)

    ax.plot(x_train, y_train, 'ko', markersize=4, label="observations")
    ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
    if add_to_plot is not None:
        add_to_plot(ax)

    plt.legend(loc=4, fontsize=15, frameon=False)
    plt.show()
    
def imshow(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
def show_rotation_on_mnist_example_image():
    mnist_train = datasets.MNIST('data', train=True, download=True)
    image = Image.fromarray(mnist_train.data[0].numpy())
    imshow(image)
    rotated_image = rotate(image, angle=90)
    imshow(rotated_image)
    
def get_mnist_data(train=True):
    mnist_data = datasets.MNIST('data', train=train, download=True)
    x = mnist_data.data.reshape(-1, 28 * 28).float()
    y = mnist_data.targets
    return x, y

def train_on_mnist(net):
    x_train, y_train = get_mnist_data(train=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    batch_size = 250

    progress_bar = trange(20)
    for _ in progress_bar:
        for batch_idx in range(int(x_train.shape[0] / batch_size)):
            batch_low, batch_high = batch_idx * batch_size, (batch_idx + 1) * batch_size
            optimizer.zero_grad()
            loss = criterion(target=y_train[batch_low:batch_high], input=net(x_train[batch_low:batch_high]))
            progress_bar.set_postfix(loss=f'{loss / batch_size:.3f}')
            loss.backward()
            optimizer.step()
    return net

def accuracy(targets, predictions):
  return (targets == predictions).sum() / targets.shape[0]

def evaluate_accuracy_on_mnist(net):
    test_data = get_mnist_data(train=False)
    x_test, y_test = test_data
    net.eval()
    y_preds = net(x_test).argmax(1)
    acc = accuracy(y_test, y_preds)
    print("Test accuracy is %.2f%%" % (acc.item() * 100))

def compute_metrics(test_score, y_test, thresh, pos_label=1):
    """
    This function compute metrics for a given threshold

    Parameters
    ----------
    test_score
    y_test
    thresh
    pos_label

    Returns
    -------

    """
    y_pred = (test_score >= thresh).astype(int)
    y_true = y_test.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, test_score)
    roc = sk_metrics.roc_auc_score(y_true, test_score)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm

def _estimate_optimal_threshold(test_score, y_test, pos_label=1, nq=100):
    # def score_recall_precision(combined_score, test_score, test_labels, pos_label=1, nq=100):
    ratio = 100 * sum(y_test == 0) / len(y_test)
    print(f"Ratio of normal data:{ratio}")
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(test_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
        # Prediction using the threshold value
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(test_score, y_test, thresh, pos_label)

        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        # print(f"qi:{qi:.3f} ==> p:{precision:.3f}  r:{recall:.3f}  f1:{f_score:.3f}")
        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }