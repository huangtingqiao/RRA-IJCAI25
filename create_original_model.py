import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
from utils.data import getdata
from utils.model import getmodel,train,test




def main(dataset,model_style,epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    trainloader ,testloader =getdata(dataset)
    model=getmodel(dataset,model_style)
    root_path = 'Original_models/' +dataset+'_'+model_style+'_epochs'+str(epochs)+'_3'
  
    if not os.path.exists(root_path):
        os.makedirs(root_path)



    test_results = []  

    train_losses = []

    if model_style=='cnn':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:    
        optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=1e-4)
           
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train(epoch, model, trainloader, device, optimizer, criterion,root_path)


        test_acc = test(model, testloader, device)
        test_results.append(test_acc)
        print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {test_acc:.2f}%')



    torch.save(model.state_dict(), root_path + '/simple_cnn_model.pth')


    with open(root_path +'/test_results.txt', 'w') as f:
        for epoch, acc in enumerate(test_results):
            f.write(f'Epoch {epoch+1}: Test Accuracy = {acc:.2f}%\n')




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model with specified dataset and model style.")

  
    parser.add_argument('--dataset', type=str, default=' cifar10', help='The dataset to use, e.g., cifar100')
    parser.add_argument('--model_style', type=str, default='cnn ', help='The model style, e.g., vgg16')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')


    args = parser.parse_args()


    main(args.dataset, args.model_style, args.epochs)