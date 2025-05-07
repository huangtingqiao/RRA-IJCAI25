import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import vgg16

class SimpleCNN(nn.Module):
    def __init__(self,num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*6*6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64*6*6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SimpleCNN_Mnist(nn.Module):
    def __init__(self,num_classes):
        super(SimpleCNN_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*5*5, 128)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        
        self.conv1=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1=nn.ReLU(inplace=True)
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2=nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.relu2=nn.ReLU(inplace=True)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3=nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3=nn.ReLU(inplace=True)

        self.conv4=nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4=nn.ReLU(inplace=True)

        self.conv5=nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5=nn.ReLU(inplace=True)
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1_pool = self.pool1(x1)
        

        x2 = self.conv2(x1_pool)
        x2 = self.relu2(x2)
        x2_pool = self.pool2(x2)
        

        x3 = self.conv3(x2_pool)
        x3 = self.relu3(x3)
        

        x4 = self.conv4(x3)
        x4 = self.relu4(x4)

        x5 = self.conv5(x4)
        x5 = self.relu5(x5)
        x5_pool = self.pool3(x5)

        x = self.avgpool(x5_pool)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class AlexNet_MNIST(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet_MNIST, self).__init__()
      
        self.conv1=nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.relu1=nn.ReLU(inplace=True)        
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2=nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2=nn.ReLU(inplace=True)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3=nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3=nn.ReLU(inplace=True)

        self.conv4=nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4=nn.ReLU(inplace=True)
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1_pool = self.pool1(x1)
        
   
        x2 = self.conv2(x1_pool)
        x2 = self.relu2(x2)
        x2_pool = self.pool2(x2)
        

        x3 = self.conv3(x2_pool)
        x3 = self.relu3(x3)
        

        x4 = self.conv4(x3)
        x4 = self.relu4(x4)
        x4_pool = self.pool3(x4)

        x = self.avgpool(x4_pool)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class BasicBlock_MNIST(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class ResNet_Mnist(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_Mnist, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

def getmodel(dataset,model_style):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)
    if dataset == 'mnist' and model_style == 'cnn':
        model=SimpleCNN_Mnist(num_classes=10).to(device)     
    elif dataset == 'mnist' and model_style == 'alexnet':
        model=AlexNet_MNIST(num_classes=10).to(device)     
    elif dataset == 'mnist' and model_style == 'resnet':
        model=ResNet_Mnist(BasicBlock_MNIST, [2, 2, 2],num_classes=10).to(device)
    elif dataset == 'mnist' and model_style == 'vgg16':
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)  
        model = model.to(device)

    elif dataset == 'cifar10'and model_style == 'cnn':
        model=SimpleCNN(num_classes=10).to(device)
    elif dataset == 'cifar10'and model_style == 'alexnet':
        model=AlexNet(num_classes=10).to(device)
    elif dataset == 'cifar10' and model_style == 'resnet':
        model=ResNet(BasicBlock, [2, 2, 2, 2],num_classes=10).to(device)   
    elif dataset == 'cifar10' and model_style == 'resnet-50':
        model=ResNet(BasicBlock, [3, 4, 6,3],num_classes=10).to(device)   
    elif dataset == 'cifar10' and model_style == 'vgg16':
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)  
        model = model.to(device)



    elif dataset == 'cifar100'and model_style == 'cnn':
        model=SimpleCNN(num_classes=100).to(device)
    elif dataset == 'cifar100'and model_style == 'alexnet':
        model=AlexNet(num_classes=100).to(device)
    elif dataset == 'cifar100' and model_style == 'resnet':
        model=ResNet(BasicBlock, [2, 2, 2, 2],num_classes=100).to(device)   
    elif dataset == 'cifar100' and model_style == 'vgg16':
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 100) 
        model = model.to(device)

    elif dataset == 'fmnist'and model_style == 'cnn':
        model=SimpleCNN_Mnist(num_classes=10).to(device)
    elif dataset == 'fmnist'and model_style == 'alexnet':
        model=AlexNet_MNIST(num_classes=10).to(device)
    elif dataset == 'fmnist' and model_style == 'resnet':
        model=ResNet_Mnist(BasicBlock_MNIST, [2, 2, 2],num_classes=10).to(device)
    elif dataset == 'cifar10' and model_style == 'vgg16':
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)  
        model = model.to(device)

    elif dataset == 'mnist_rgb' and model_style == 'cnn':
        model=SimpleCNN(num_classes=10).to(device)     
    elif dataset == 'mnist_rgb' and model_style == 'alexnet':
        model=AlexNet(num_classes=10).to(device)    

    return model


def train(epoch, model, trainloader, device, optimizer, criterion,root_path):
    model.train()  
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()  
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:    
            print(f'Epoch {epoch + 1}, Batch [{batch_idx+1}/{len(trainloader)}], Loss: {running_loss / 100:.3f}')

            with open(root_path + '/train_losses.txt', 'a') as f: 
                f.write(f'Epoch {epoch + 1}, Batch [{batch_idx+1}/{len(trainloader)}], Loss: {running_loss / 100:.3f}\n')

            running_loss = 0.0  

    if running_loss > 0:
            print(f'Epoch {epoch + 1}, Batch [{batch_idx+1}/{len(trainloader)}], Loss: {running_loss / ((batch_idx + 1)%100):.3f}')
            with open(root_path + '/train_losses.txt', 'a') as f:       
                f.write(f'Epoch {epoch + 1}, Batch [{batch_idx+1}/{len(trainloader)}], Loss: {running_loss / ((batch_idx + 1)%100):.3f}\n')

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
