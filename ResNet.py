import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import sys
import numpy

print(f'torch : {torch.__version__}')
print(f'python : {sys.version}')

# Train_data transform

train_transform = transforms.Compose(
    [   
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)

batch_size = 128

# data download
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size,shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform = test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle= True)

classes = ('plane','car','bird','cat','deer','dog','frog', 'horse', 'ship', 'truck')

print(f'train_set : {len(train_set)}')
print(f'test_set  : {len(test_set)}')

class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(VGG,self).__init__()

        #self.features = features
        self.convlayer = nn.Sequential(
            # RGB 3 - > 64 / size(224,224)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size (112,112)

            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size (56, 56)

            nn.Conv2d(in_channels=128, out_channels = 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size (28, 28)

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size (14,14)

            # nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            # size (7,7)

        )
        #self.avgpool = nn.AdaptiveAvgPool2d(5)

        self.fclayer =nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )

    def forward(self, x):
        x = self.convlayer(x)
        x = torch.flatten(x,1)
        x = self.fclayer(x)
        return x

vgg11 = VGG(num_classes=10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

if device == 'cuda':
    vgg11 = vgg11.to(device)
# summary
print(vgg11)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg11.parameters(), lr = 0.0001)

loss_list = []
for epoch in range(10):
    running_loss = 0.0
    start_time = time.time()

    for i, data in enumerate(train_loader):
        inputs, labels = data
        if device =='cuda':
            inputs = inputs.to(device)
            labels = labels.to(device)
        
        # gradients to zero
        optimizer.zero_grad()

        outputs = vgg11(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            print(f'mini batch time : {time.time-start_time}')
    print(f'epoch : {epoch+1}\n loss : {running_loss/5000:.3f}, time :{time.time()-start_time}')
    loss_list.append(running_loss/128)

print(f'{"-"*20}\nFinish running\n\n')

correct = 0
total = 0
class_correct = []
class_total = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        if device =='cuda':
            images= imgaes.to(device)
            labels= labels.to(device)
        
        outputs = vgg11(images)
        _, predicted = torch.max(outputs,1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c =(predicted== labels).squeeze()
        for i in range(4):
            labels = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(f'Accuracy of {classes[i]:5s} : {100*class_correct/class_total:.1f}%')

print(f'Total Accuracy : {100* correct /total:.1f}%')