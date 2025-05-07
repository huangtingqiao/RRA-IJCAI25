import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os,copy
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


def prune_model(model, pruning_rate, output_file):
    parameters_to_prune = []
    new_model = copy.deepcopy(model) 

    for name, module in new_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(new_model.state_dict(), output_file)
    return new_model

def prune_model_entire_kernels(model, pruning_rate, output_file):

    new_model = copy.deepcopy(model)
    
    parameters_to_prune = []
    

    for name, module in new_model.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))


    for module, param in parameters_to_prune:
        prune.ln_structured(module, name=param, amount=pruning_rate, n=2, dim=0)
    

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    torch.save(new_model.state_dict(), output_file)
    
    return new_model


def load_pruned_model(model, file_path):
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))


    new_state_dict = {}
    for key, value in state_dict.items():
        if '_orig' in key:
            param_name = key.replace('_orig', '')
            mask_name = param_name + '_mask'
            masked_value = value * state_dict[mask_name]
            new_state_dict[param_name] = masked_value
        elif '_mask' in key:
            continue 
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified dataset and model style.")

  
    parser.add_argument('--dataset', type=str, default=' cifar10', help='The dataset to use, e.g., cifar100')
    parser.add_argument('--model_style', type=str, default='alexnet ', help='The model style, e.g., vgg16')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
 


    args = parser.parse_args()

    dataset=args.dataset
    model_style=args.model_style
    epochs=args.epochs


    root_path = 'FP/'+ dataset+'_'+model_style+'_epochs'+str(epochs)

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    model_path='Original_models/' +dataset+'_'+model_style+'_epochs'+str(epochs)+'/simple_cnn_model.pth'
    model=getmodel(dataset,model_style)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pruning_rates = [ 1/16,2/16,3/16,4/16]


    trainloader, testloader = getdata(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = test(model, testloader, device)
    print(f'before pruning Test accuracy: {test_acc}')


    for rate in pruning_rates:

        pruned_model = prune_model_entire_kernels(model, rate, root_path + f'/model/pruned_model_{rate:.2f}.pth')
        
        pruned_model.eval()
        

        test_acc = test(pruned_model, testloader, device)
        if model_style=='vgg16': 
            dct_filters_conv1 = extract_and_dct_conv_filters(pruned_model.features[0], root_path +f'/dct/{rate:.2f}_conv1_filters_dct.txt')
        else:
            dct_filters_conv1 = extract_and_dct_conv_filters(pruned_model.conv1, root_path +f'/dct/{rate:.2f}_conv1_filters_dct.txt')


        print(f'Pruning rate: {rate}, Test accuracy: {test_acc}')
        with open(root_path  +'/ACC.txt', 'a') as f:
            f.write(f'{rate:.2f}Prue Test accuracy: {test_acc:.2f}\n%')
    if model_style=='vgg16': 
        ora_dct_filters_conv1 = extract_and_dct_conv_filters(model.features[0], root_path +f'/ora_conv1_filters_dct.txt')
    else:
        ora_dct_filters_conv1 = extract_and_dct_conv_filters(model.conv1, root_path +f'/ora_conv1_filters_dct.txt')

    prune_ratios = [1/16,2/16,3/16,4/16]  
    prue_model = getmodel(dataset, model_style)

    for ratio in prune_ratios:
        model_path = f'{root_path}/model/pruned_model_{ratio:.2f}.pth'
        prue_model = load_pruned_model(prue_model, model_path)
        if model_style=='vgg16': 
            prue_dct_filters_conv1 = extract_and_dct_conv_filters(prue_model.features[0], root_path + f'/prue_conv1_filters_dct_{ratio:.2f}.txt')

        else:
            prue_dct_filters_conv1 = extract_and_dct_conv_filters(prue_model.conv1, root_path + f'/prue_conv1_filters_dct_{ratio:.2f}.txt')

        compare_path = f"{root_path}/compare/{ratio:.2f}"
        if not os.path.exists(compare_path):
            os.makedirs(compare_path)

        distances1_2=compare_cossim_dct_filters_long(ora_dct_filters_conv1,prue_dct_filters_conv1)
  
        for idx, score in enumerate(distances1_2):
            print(f" FS : {1-score}")
        with open(compare_path  +f'/FS.txt', 'w') as f:
            for idx, score in enumerate(distances1_2):
                f.write(f'distances for dct filter {idx + 1}: {score}\n')
        with open(compare_path  +f'/FS.txt', 'a') as f:
            for idx, score in enumerate(distances1_2):
                f.write(f'FS : {1-score}\n')
        



  