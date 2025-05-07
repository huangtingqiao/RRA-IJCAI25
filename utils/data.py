from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import transforms


import torch


trans_mnist = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

trans_cifar10_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar10_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trans_cifar100_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)
trans_cifar100_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)


trans_fashion_mnist = transforms.Compose(
    [
        transforms.Resize((32, 32)), 
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


trans_imagenet_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trans_imagenet_val = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
transform_mnist_RGB = transforms.Compose(
    [
        transforms.Resize((32, 32)), 
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


transform_imagenet = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def getdata(dataset):
    if dataset == "mnist":
        dataset_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=trans_mnist
        )
        dataset_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=trans_mnist
        )

    elif dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=trans_cifar10_train
        )
        dataset_test = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=trans_cifar10_val
        )
   
    elif dataset == "cifar100":
        dataset_train = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=trans_cifar100_train
        )
        dataset_test = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=trans_cifar100_val
        )
    elif dataset == "fmnist":
        dataset_train = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=trans_mnist
        )
        dataset_test = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=trans_mnist
        )
    elif dataset == "mnist_rgb":
        dataset_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist_RGB
        )
        dataset_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist_RGB
        )


    trainloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=64, shuffle=True)
    return trainloader, testloader
