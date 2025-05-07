from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import getdata
from utils.gren_dct import extract_and_dct_conv_filters
import copy,os
from utils.model import test,getmodel,train
from utils.compare_cos import compare_cossim_dct_filters_long

transform_mnist_1D_3D = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mnist_3d_train_dataset = MNIST(root='./data', train=True, download=True, transform=transform_mnist_1D_3D )
mnist_3d_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_mnist_1D_3D )

mnist_3d_train_loader = DataLoader(mnist_3d_train_dataset, batch_size=64, shuffle=True)
mnist_3d_test_loader = DataLoader(mnist_3d_test_dataset, batch_size=64, shuffle=False)

def main(ora_dataset,model_style,epochs,tuning_dataset,ft_epochs):    
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path='Original_models/' +ora_dataset+'_'+model_style+'_epochs'+str(epochs)+'/simple_cnn_model.pth'
    model=getmodel(ora_dataset,model_style)
    model.load_state_dict(torch.load(model_path)) 

    root_path = 'fine_tuning_retrain/'+model_style+ ora_dataset+'_to_'+tuning_dataset+'_epochs'+str(epochs)+'_ft_epochs'+str(ft_epochs)

    if not os.path.exists(root_path):
        os.makedirs(root_path)


    ft_model = copy.deepcopy(model) 

    ft_model.to(device)

    ora_trainloader, ora_testloader = getdata(ora_dataset)
    ora_before_test_acc = test(model, ora_testloader, device)
    print(f'Init Test accuracy: {ora_before_test_acc:.2f}%')
    with open(root_path  +'/ACC.txt', 'w') as f:
        f.write(f'Oradataset Init Test accuracy: {ora_before_test_acc:.2f}%\n')
   
    ft_trainloader, ft_testloader = getdata(tuning_dataset)
    ft_before_test_acc = test(ft_model, ft_testloader, device)
    print(f'Init ft Test accuracy: {ft_before_test_acc:.2f}%')
    with open(root_path  +'/ACC.txt', 'a') as f:
        f.write(f'tuning dataset Init Test accuracy: {ft_before_test_acc:.2f}%\n')
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ft_model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(ft_epochs):  
        ft_model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(ft_trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = ft_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:   
                print(f'Epoch {epoch + 1}, Batch [{batch_idx+1}/{len(ft_trainloader)}], Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    with open(root_path  +'/ACC.txt', 'a') as f:
        f.write(f'\n')

    after_ft_model = copy.deepcopy(ft_model) 
    

    after_ft_model.to(device)
    oraset_afterfine_test_acc = test(after_ft_model, ora_testloader, device)
    print(f'Init Test accuracy: {oraset_afterfine_test_acc:.2f}%')
    with open(root_path  +'/ACC.txt', 'a') as f:
        f.write(f'Oradataset after fine Test accuracy: {oraset_afterfine_test_acc:.2f}%\n')


    ft_before_test_acc = test(ft_model, ft_testloader, device)
    print(f'Init ft Test accuracy: {ft_before_test_acc:.2f}%')
    with open(root_path  +'/ACC.txt', 'a') as f:
        f.write(f'Tuningdataset after fine Test accuracy: {ft_before_test_acc:.2f}%\n')
    

    torch.save(ft_model.state_dict(), root_path+'/fine_tuning_model_filters.pth')

    if model_style=='vgg16':      
         ora_dct_filters_conv1 = extract_and_dct_conv_filters(model.features[0], root_path +f'/ora_conv1_filters_dct.txt')
    else:    
        ora_dct_filters_conv1 = extract_and_dct_conv_filters(model.conv1, root_path +f'/ora_conv1_filters_dct.txt')

  
    if model_style=='vgg16':      
        ft_dct_filters_conv1 = extract_and_dct_conv_filters(ft_model.features[0], root_path +f'/ft_conv1_filters_dct.txt')
    else:
        ft_dct_filters_conv1 = extract_and_dct_conv_filters(ft_model.conv1, root_path +f'/ft_conv1_filters_dct.txt')


    compare_path=root_path+'/compare' 
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)



    distances1_2 =compare_cossim_dct_filters_long(ora_dct_filters_conv1,ft_dct_filters_conv1)


    for idx, score in enumerate(distances1_2):
        print(f" FS : {1-score}")
    with open(compare_path  +f'/FS.txt','w') as f:
        for idx, score in enumerate(distances1_2):
            f.write(f'distances for dct filter {idx + 1}: {score}\n')
    with open(compare_path  +f'/FS.txt', 'a') as f:
        for idx, score in enumerate(distances1_2):
            f.write(f'FS : {1-score}\n')
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified dataset and model style.")

  
    parser.add_argument('--ora_dataset', type=str, default=' cifar10', help='The original dataset to use, e.g., cifar10')
    parser.add_argument('--model_style', type=str, default='alexnet ', help='The model style, e.g., vgg16')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--ft_dataset', type=str, default=' cifar10', help='The fine-tuning dataset to use, e.g., cifar10')
    parser.add_argument('--ft_epochs', type=int, default=20, help='Number of fine_tuning epochs for training')

    args = parser.parse_args()

    ora_dataset=args.ora_dataset
    ft_dataset=args.ft_dataset
    model_style=args.model_style
    epochs=args.epochs
    ft_epochs=args.ft_epochs
 

    main(ora_dataset,model_style,epochs,ft_dataset,ft_epochs)


