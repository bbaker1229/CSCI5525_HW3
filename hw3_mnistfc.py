import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128, bias=True)
        self.rlu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.soft(self.fc2(x))
        return x


def get_mnist_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train, test


def load_mnist_data(train, test):
    train_load = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
    test_load = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False, num_workers=0)
    return train_load, test_load


train_set, test_set = get_mnist_data()
train_loader, test_loader = load_mnist_data(train_set, test_set)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

training_loss = []
test_loss = []
for epoch in range(30):
    print('Epoch #: %d' % (epoch + 1))
    running_loss = 0.0
    total = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parmeter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total += 1.0
        # if i % 500 == 499:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 500))
        #     running_loss = 0.0
    training_loss.append(running_loss / total)
    # print(training_loss)
    print('Training Loss: %.4f' % (
            training_loss[epoch]))
    running_loss = 0.0
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            # _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
            total += 1.0
            # correct += (predicted == labels).sum().item()
    test_loss.append(running_loss / total)
    # print(test_loss)
    print('Testing Loss: %.4f' % (
            test_loss[epoch]))

print("Finished Training")

