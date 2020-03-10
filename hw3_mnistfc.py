############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
# File Name: hw3_mnistfc.py
# Saved Neural Network File Name: mnist-fc
############################################


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# Define the Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.soft(self.fc2(x))
        return x


def get_mnist_data():
    """
    Get the MNIST dataset from torchvision.
    :return: Train and test data sets.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train, test


def load_mnist_data(train, test):
    """
    Create training and test dataset loaders.
    :param train: Training dataset.
    :param test: Test dataset.
    :return: training and test data loaders.
    """
    train_load = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
    test_load = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False, num_workers=0)
    return train_load, test_load


# Get data set and prepare data loaders for torch
train_set, test_set = get_mnist_data()
train_loader, test_loader = load_mnist_data(train_set, test_set)

# Set up Neural Network
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# Start Neural Network training on 30 Epochs
# Record the batch loss and the training accuracy for each epoch
training_loss = []
training_acc = []
for epoch in range(500):
    print('Epoch #: %d' % (epoch + 1))
    running_loss = 0.0
    batch_total = 0.0
    image_total = 0.0
    correct = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        image_total += labels.size(0)
        batch_total += 1.0
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    training_loss.append(running_loss / batch_total)
    training_acc.append(100 * correct / image_total)
    # print(training_loss)
    print('Training Loss: %.4f' % (
            training_loss[epoch]))
    print('Training Accuracy: %.4f %%' % (
            training_acc[epoch]))
    if (epoch > 0) and ((training_acc[epoch] - training_acc[epoch - 1]) < 0.01):
        break
print("Finished Training")

# Plot Training Loss
plt.plot(training_loss)
plt.title("MNIST Data Training loss on A Fully - Connected Neural Network")
plt.ylabel("Cross Entropy Training Loss")
plt.xlabel("Epoch")
plt.show()

# Plot Training Accuracy
plt.plot(training_acc)
plt.title("MNIST Data Training Accuracy on A Fully - Connected Neural Network")
plt.ylabel("Training Accuracy")
plt.xlabel("Epoch")
plt.show()

# Save the trained neural network
PATH = './mnist-fc'
torch.save(net.state_dict(), PATH)

# Reload the neural network
net = Net()
net.load_state_dict(torch.load(PATH))

# Determine the accuracy of the test dataset
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test dataset: %d %%' % (
    100 * correct / total))

# Print accuracies for each class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))
