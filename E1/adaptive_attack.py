import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dct
import random
import argparse
import torch.nn.functional as F
from scipy.fftpack import dctn, idctn
import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data import getdata
from utils.model import getmodel,train,test
from utils.compare_cos import compare_cossim_dct_filters_long



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_and_dct_conv_filters(layer, output_file):
    filters = layer.weight.data.cpu().numpy()
    dct_filters = []
    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)
    with open(output_file, 'w') as f:
        for filter_idx in range(filters.shape[0]):
            filter_data = filters[filter_idx]
            f.write(f'Convolutional Filter {filter_idx + 1} (Before DCT):\n')
            f.write(np.array2string(filter_data, separator=',') + '\n')

            filter_dct = dctn(filter_data, type=2, norm='ortho')  
            f.write(f'Convolutional Filter {filter_idx + 1} (After DCT):\n')
            f.write(np.array2string(filter_dct, separator=',') + '\n\n')
            dct_filters.append(filter_dct)

    return dct_filters



def modify_and_save_dct_filters(dct_filters, output_file,rate1,rate2,percentage_to_modify):
    modified_dct_filters = []
    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)
    with open(output_file, 'w') as f:
        for filter_idx, filter_dct in enumerate(dct_filters):
            depth, height, width = filter_dct.shape
            total_elements = depth * height * width

            threshold1 = (depth + height + width - 3) * rate1
            threshold2 = (depth + height + width - 3) * rate2


            extracted_data = []
            for d in range(depth):
                for h in range(height):
                    for w in range(width):
                        if rate1==1/3 and rate2== 2/3:
                            if threshold1<d + h + w < threshold2:
                                extracted_data.append((d, h, w))
                       
                        elif threshold1<=d + h + w <= threshold2:
                            extracted_data.append((d, h, w))


            num_to_modify = int(len(extracted_data) * percentage_to_modify)
            indices_to_modify = random.sample(range(len(extracted_data)), num_to_modify)


            for index in indices_to_modify:
               
                d, h, w = extracted_data[index] 
                filter_dct[d, h, w] = random.random()


            f.write(f'Modified Convolutional Filter {filter_idx + 1} (After Modification):\n')
            f.write(np.array2string(filter_dct, separator=',') + '\n\n')

            modified_dct_filters.append(filter_dct)

    return modified_dct_filters


def modify_idct_filters(dct_filters, output_file):
    new_filters_conv = []
    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)
    with open(output_file, 'w') as f:
        for idx, filter_dct in enumerate(dct_filters):
            inv_filter = idctn(filter_dct, type=2, norm='ortho') 
            new_filters_conv.append(inv_filter)

            f.write(f'Convolutional Filter {idx + 1}:\n')
            f.write(np.array2string(inv_filter, separator=',') + '\n\n')
    
    return new_filters_conv


def inverse_dct_and_save(conv_layers_weights, original_model, output_file):
    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)
    new_model = copy.deepcopy(original_model)  
    
    for layer_name, new_weights in conv_layers_weights.items():
        if model_style=='vgg16':
            layer=new_model.features[0]
        else:
            layer = getattr(new_model, layer_name)  
        layer.weight.data = torch.tensor(np.array(new_weights))


    torch.save(new_model.state_dict(), output_file)
    return new_model


def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total * 100

def main(dataset,model,model_style,epochs,root_path,num_layers,filter_names,rate1,rate2,percentage_to_modify):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    trainloader ,testloader =getdata(dataset)
    test_acc = test(model, testloader, device)
    with open(root_path  +'/ACC.txt', 'w') as f:
        f.write(f'Original test accuracy: {test_acc:.2f}%\n')

    conv_layers_weights = {}
    dct_filters_list = []  
    if model_style=='vgg16':
        conv_layers =[model.features[0]]
    else:
        conv_layers = [getattr(model, f'conv{i+1}') for i in range(num_layers)]
    
    for i, conv_layer in enumerate(conv_layers):

        dct_file_path = os.path.join(root_path, 'ora', f'{filter_names[i]}_filters_dct.txt')
        modified_file_path = os.path.join(root_path, 'mod', f'dac_modified_{filter_names[i]}_filters_dct.txt')
        new_filters_file_path = os.path.join(root_path,'mod',  f'filt_modified_{filter_names[i]}_filters.txt')
        

        dct_filters = extract_and_dct_conv_filters(conv_layer, dct_file_path)
        dct_filters_list.append(dct_filters)

        dct_filters_copy = copy.deepcopy(dct_filters)
        modified_dct_filters = modify_and_save_dct_filters(dct_filters_copy , modified_file_path,rate1,rate2,percentage_to_modify)
        

        new_filters = modify_idct_filters(modified_dct_filters, new_filters_file_path)
        conv_layers_weights[filter_names[i]] = new_filters

    new_model=inverse_dct_and_save(conv_layers_weights, model, root_path+'/new_model.pth')
    new_model.to(device)

    new_dct_filters_list = [] 
    if model_style=='vgg16':
        conv_layers =[new_model.features[0]]
    else:
        conv_layers = [getattr(new_model, f'conv{i+1}') for i in range(num_layers)]
    for i, conv_layer in enumerate(conv_layers):

        dct_file_path = os.path.join(root_path, 'new', f'{filter_names[i]}_filters_dct.txt')
        

        new_dct_filters = extract_and_dct_conv_filters(conv_layer, dct_file_path)
        new_dct_filters_list.append(new_dct_filters)

    test_acc = test(new_model, testloader, device)

    with open(root_path  +'/ACC.txt', 'a') as f:
        f.write(f'attack accuracy: {test_acc:.2f}%\n')

 
    compare_path=root_path+'/compare' 
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)

    for i, conv_layer in enumerate(conv_layers):



        distances1_2=compare_cossim_dct_filters_long(dct_filters_list[i],new_dct_filters_list[i],rate3)




        for idx, score in enumerate(distances1_2):
            print(f" FS : {score}")
        with open(compare_path  +'/FS.txt', 'w') as f:
            for idx, score in enumerate(distances1_2):
                f.write(f'distances for dct filter {idx + 1}: {score}\n')
        with open(compare_path  +'/FS.txt', 'w') as f:
            for idx, score in enumerate(distances1_2):
                f.write(f'FS : {1-score}\n')
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified dataset and model style.")  
    parser.add_argument('--dataset', type=str, default=' cifar10', help='The dataset to use, e.g., cifar100')
    parser.add_argument('--model_style', type=str, default='alexnet ', help='The model style, e.g., vgg16')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')

    args = parser.parse_args()

    dataset=args.dataset
    model_style=args.model_style
    epochs=args.epochs

    rate1=0
    rate2=2/3
    num_layers=1
    rate3=2/3
    percentage_to_modify_list = [1/10, 2/10, 3/10,4/10]  
    epochsss = [1, 2, 3, 4, 5]

    def format_rate(rate):
        if rate == 1 / 3:
            return '1-3'
        elif rate == 2 / 3:
            return '2-3'
        return str(rate)

    for epochss in epochsss:
        for percentage_to_modify in percentage_to_modify_list:  
            root_path = f'Adaptive_attack/{dataset}_{model_style}_epochs{epochs}/{format_rate(rate1)}_{format_rate(rate2)}_{percentage_to_modify}_{epochss}'

            if not os.path.exists(root_path):
                os.makedirs(root_path)

            model_path = f'Original_models/{dataset}_{model_style}_epochs{epochs}/simple_cnn_model.pth'
            model = getmodel(dataset, model_style)
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            filter_names = [f'conv{i + 1}' for i in range(num_layers)]

            main(dataset, model, model_style, epochs, root_path, num_layers, filter_names, rate1, rate2, percentage_to_modify)
