from utils import train, evaluate
from pathlib import Path
from metrics import confusion_matrix_metrics, effect_size, inversion_number
from metrics import friedman_test_for_4_samples, friedman_test_for_8_samples
from plots import heatmap, data_set_distribution, redm, training_loss
import argparse
from data_processing import process_raw_data, split_and_save_data


directory_model = "checkpoints/"
directory_data = "data/"
directory_output = "outputs/"

batch_size = 32
lr = 1e-5
w_d = 1e-5        
momentum = 0.9   
epochs = 20
is_train = False
to_process_data = False

def init():
    Path(directory_model).mkdir(parents=True, exist_ok=True)
    Path(directory_data).mkdir(parents=True, exist_ok=True)
    Path(directory_output).mkdir(parents=True, exist_ok=True)

def run(batch_size, lr, w_d, momentum, epochs, is_train, to_process_data):
    
    if to_process_data:
        process_raw_data()
        split_and_save_data()
        
    if is_train:
        init()
        train(batch_size, lr, w_d, momentum, epochs)
    evaluate()

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="CUQ-AE-REDM Framework for uncertainty quantification on AEs-based methods for anomaly detection",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument("-t", "--is_train", action="store_true", help="training mode", default=False)
   parser.add_argument("-p", "--to_process_data", action="store_true", help="process data", default=False)
   parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=32) 
   parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-5) 
   parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-5) 
   parser.add_argument('-m', '--momentum', nargs='?', const=1, type=float, default=0.9) 
   parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=20) 
   #parser.add_argument("src", help="Source location")
   args = parser.parse_args()
   configs = vars(args)
   is_train = configs['is_train']
   to_process_data = configs['to_process_data']
   batch_size = configs['batch_size']
   lr = configs['learning_rate']
   w_d = configs['weight_decay']       
   momentum = configs['momentum']
   epochs = configs['epochs']
   run(batch_size, lr, w_d, momentum, epochs, is_train, to_process_data)
