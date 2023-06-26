from utils import train, evaluate
from pathlib import Path

directory_model = "checkpoints/"
directory_data = "data/"
directory_output = "outputs/"

batch_size = 32
lr = 1e-5
w_d = 1e-5        
momentum = 0.9   
epochs = 5

def init():
    Path(directory_model).mkdir(parents=True, exist_ok=True)
    Path(directory_data).mkdir(parents=True, exist_ok=True)
    Path(directory_output).mkdir(parents=True, exist_ok=True)

def run(is_train = True):
    
    if is_train:
        train(batch_size, lr, w_d, momentum, epochs)
    else:
        evaluate()
    
def results():
    