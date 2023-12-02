# references https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#            https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch

import torchvision
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import date
import os

from sklearn.metrics import confusion_matrix

# define transformation for each data set
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# separate data
data_dir = 'images'
train_data = datasets.ImageFolder(os.path.join(data_dir, "train"),train_transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, "test"),test_transform)

# define data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True, num_workers=4)

class_names = train_data.classes


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

class WaterBirdCNNet(nn.Module):
    def __init__(self):
        super(WaterBirdCNNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 8)
        self.conv2 = nn.Conv2d(48, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 640)
        self.fc2 = nn.Linear(640,120)
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def trainModel(net,trainloader,criterion,optimizer,epoch=2):
    for epoch in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # convert it to CUDA tensors
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

def predict(model,path):
    img = Image.open(path)
    t_img = test_transform(img).unsqueeze(0)
    t_img = t_img.to(device)
    output = model(t_img)
    _, prediction = torch.max(output, 1)
    #print("result = " + class_names[prediction[0].item()])
    #img.show()
    plt.imshow(img)
    plt.title("result = " + class_names[prediction[0].item()])
    plt.show()

if __name__ == '__main__':
    # use CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Show some samples
    #inputs, classes = next(iter(train_loader))
    #out = torchvision.utils.make_grid(inputs)
    #imshow(out, title=[class_names[x] for x in classes])

    # construct Network
    net = WaterBirdCNNet()
    net.to(device)
    
    # define optimiser and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # training
    start = time.time()
    trainModel(net,train_loader,criterion,optimizer,3)
    process_time = time.time() - start
    print(process_time)

    # save model's infromations
    PATH = "./waterBird_net_"+str(date.today())+".pth"
    torch.save(net.state_dict(), PATH)
    
    # load the information
    net = WaterBirdCNNet()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    # test Accuracy of the model
    
    correct = 0
    total = 0
    with torch.no_grad():
        prediction = []
        test_labels = []
        for data in test_loader:
            images, labels = data
            inputs = images
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prediction += [p.item() for p in predicted]
            test_labels += [l.item() for l in labels]
            # show predictions
            out = torchvision.utils.make_grid(inputs)
            imshow(out, title=[class_names[x] for x in predicted])
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    m = confusion_matrix(test_labels,prediction)
    print("Confusion Matrix:")
    print(m)
    print("Bird predicted as Bird    : ",m[0][0])
    print("Bird predicted as Trash   : ",m[0][1])
    print("Trash predicted as Trash  : ",m[1][1])
    print("Trash predicted as Bird   : ",m[1][0])

