    
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os,copy,time
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model import SimpleCNN,SimpleCNN_Mnist
from utils.data import getdata
import numpy as np
import argparse
from scipy.fftpack import dctn, idctn
from utils.gren_dct import extract_and_dct_conv_filters


from utils.model import getmodel,train,test
from utils.compare_cos import compare_cossim_dct_filters_long


def main(model1,model2,model_style,root_path,time_file_path):

    compare_path=root_path+'/compare' 
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)
    start_times=time.time()
    if model_style=='vgg16':
        model1conv_layers =model1.features[0]
    else:
        model1conv_layers =model1.conv1
    start_times=time.time()
    model1_dct_filters_conv1 = extract_and_dct_conv_filters(model1conv_layers, root_path +f'/dct/model1_conv1_filters_dct.txt')

    end_times=time.time()            
    elapsed_times=end_times-start_times
    with open(time_file_path, 'a') as time_file:
        result_line = f"Gen. time: {elapsed_times:.6f}\n"
        print(result_line)
        time_file.write(result_line) 

    start_times2=time.time()
    if model_style=='vgg16':
        model2conv_layers =model2.features[0]
    else:
        model2conv_layers =model2.conv1
    
    model2_dct_filters_conv1 = extract_and_dct_conv_filters(model2conv_layers, root_path +f'/dct/model2_conv1_filters_dct.txt')

    distances1_2 =compare_cossim_dct_filters_long(model1_dct_filters_conv1,model2_dct_filters_conv1)


    end_times2=time.time()            
    elapsed_times2=end_times2-start_times2
    with open(time_file_path, 'a') as time_file:
        result_line = f"Ver. time: {elapsed_times2:.6f}\n"
        print(result_line)
        time_file.write(result_line)  


            
    for idx, score in enumerate(distances1_2):
        print(f" FS {idx + 1}: {1-score}")
    with open(compare_path  +'/FS.txt', 'w') as f:
        for idx, score in enumerate(distances1_2):
            f.write(f'distances for dct filter {idx + 1}: {score}\n')
    with open(compare_path  +'/FS.txt', 'w') as f:
        for idx, score in enumerate(distances1_2):
            f.write(f'FS {idx + 1}: {1-score}\n')
    




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Train a model with specified dataset and model style.")  
    parser.add_argument('--dataset1', type=str, default=' cifar10', help='The dataset to use, e.g., cifar100')
    parser.add_argument('--dataset2', type=str, default=' cifar10', help='The dataset to use, e.g., cifar100')
    parser.add_argument('--model_style1', type=str, default='alexnet ', help='The model style, e.g., vgg16')
    parser.add_argument('--model_style2', type=str, default='alexnet ', help='The model style, e.g., vgg16')
    parser.add_argument('--epochs1', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--epochs2', type=int, default=50, help='Number of epochs for training')


    args = parser.parse_args()

    dataset1=args.dataset1
    model_style1=args.model_style1
    epochs1=args.epochs1

    dataset2=args.dataset2
    model_style2=args.model_style2
    epochs2=args.epochs2


    root_path = 'time_use/'+ dataset1+'_'+model_style1+'_epochs'+str(epochs1)+ '_'+dataset2+'_'+model_style2+'_epochs'+str(epochs2)


    if not os.path.exists(root_path):
        os.makedirs(root_path)
    time_file_path = os.path.join(root_path, 'time_results.txt')

    os.makedirs(os.path.dirname(time_file_path), exist_ok=True)
    model1_path='Original_models/' +dataset1+'_'+model_style1+'_epochs'+str(epochs2)+'/simple_cnn_model.pth'

    model2_path='Original_models/' +dataset2+'_'+model_style2+'_epochs'+str(epochs2)+'_2/simple_cnn_model.pth'

    model1=getmodel(dataset1,model_style1)
    model2=getmodel(dataset2,model_style2)
    model1.load_state_dict(torch.load(model1_path))
    model2.load_state_dict(torch.load(model2_path))
    model2.eval()
    main(model1,model2,model_style1,root_path,time_file_path)