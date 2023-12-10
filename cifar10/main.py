import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime



def run_all(learning_rate,momentum):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # convolutional layer
            ##in,out,kernel size,padding
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.conv3 = nn.Conv2d(64, 128, 3)
            # pool layer
            self.pool = nn.MaxPool2d(2, 2)
            # fully connected layer
            self.fc1 = nn.Linear(128 * 2 * 2, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 10)
            # dropout layer
            ## Parameter p indicates the probability of a unit (neuron) being disabled. 
            ## Probability of a unit being disabled in each training iteration is 20% (0.2).
            self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.dropout1(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.dropout1(x)
            x = self.pool(F.relu(self.conv3(x)))
            x = self.dropout1(x)
            # flattening(prepare image data for fully connected data input)
            x = x.view(-1, 128 * 2 * 2)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x) #output layer

            return x
    data_dir = './dataset'
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=2)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = Net()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(momentum, momentum))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    epoch_losses = []
    training_time_start = datetime.now()
    model.train()
    for epoch in range(20):
        running_loss = 0.0
        saved_loss = 0.0
        for i, data in enumerate(train_loader, 0):           
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('%d, %5d| loss: %.3f' %(epoch+1, i+1, running_loss/2000))
                saved_loss = running_loss
                running_loss = 0.0
        epoch_losses.append(saved_loss/10000)
    training_time_end = datetime.now()
    epochs = range(1,21)
    plt.plot(epochs, epoch_losses, 'g', label='Training loss')
    plt.title('Trainingloss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()    
    training_time = training_time_end - training_time_start
    print('Training done!')
    total = 0
    correct = 0
    test_time_start = datetime.now()
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_time_end = datetime.now()
    test_time = test_time_end - test_time_start
    test_accuracy= (100 * correct / total)
    print('Accuracy: %d %%' % (100 * correct / total))
    f = open("rms.txt", "a")
    f.write("\n learning rate: " + str(learning_rate) + 
            " momentum: " + str(momentum) + 
            " training loss" + str(sum(epoch_losses)/len(epoch_losses)) + 
            " test accuracy: " + str(test_accuracy) + 
            " training time: " + str(training_time) + 
            " test time: " + str(test_time)
            )
    f.close()

def main():
    run_all(learning_rate=0.001,momentum=0.9)
    run_all(learning_rate=0.001,momentum=0.6)
    run_all(learning_rate=0.001,momentum=0.3)
    run_all(learning_rate=0.001,momentum=0.1)

    run_all(learning_rate=0.003,momentum=0.9)
    run_all(learning_rate=0.003,momentum=0.6)
    run_all(learning_rate=0.003,momentum=0.3)
    run_all(learning_rate=0.003,momentum=0.1)

    run_all(learning_rate=0.005,momentum=0.9)
    run_all(learning_rate=0.005,momentum=0.6)
    run_all(learning_rate=0.005,momentum=0.3)
    run_all(learning_rate=0.005,momentum=0.1)

    run_all(learning_rate=0.007,momentum=0.9)
    run_all(learning_rate=0.007,momentum=0.6)
    run_all(learning_rate=0.007,momentum=0.3)
    run_all(learning_rate=0.007,momentum=0.1)

    run_all(learning_rate=0.009,momentum=0.9)
    run_all(learning_rate=0.009,momentum=0.6)
    run_all(learning_rate=0.009,momentum=0.3)
    run_all(learning_rate=0.009,momentum=0.1)

    run_all(learning_rate=0.0001,momentum=0.9)
    run_all(learning_rate=0.0001,momentum=0.6)
    run_all(learning_rate=0.0001,momentum=0.3)
    run_all(learning_rate=0.0001,momentum=0.1)

    run_all(learning_rate=0.0003,momentum=0.9)
    run_all(learning_rate=0.0003,momentum=0.6)
    run_all(learning_rate=0.0003,momentum=0.3)
    run_all(learning_rate=0.0003,momentum=0.1)

    run_all(learning_rate=0.0005,momentum=0.9)
    run_all(learning_rate=0.0005,momentum=0.6)
    run_all(learning_rate=0.0005,momentum=0.3)
    run_all(learning_rate=0.0005,momentum=0.1)


    run_all(learning_rate=0.0007,momentum=0.9)
    run_all(learning_rate=0.0007,momentum=0.6)
    run_all(learning_rate=0.0007,momentum=0.3)
    run_all(learning_rate=0.0007,momentum=0.1)


    run_all(learning_rate=0.0009,momentum=0.9)
    run_all(learning_rate=0.0009,momentum=0.6)
    run_all(learning_rate=0.0009,momentum=0.3)
    run_all(learning_rate=0.0009,momentum=0.1)

    run_all(learning_rate=0.01,momentum=0.9)
    run_all(learning_rate=0.01,momentum=0.6)
    run_all(learning_rate=0.01,momentum=0.3)
    run_all(learning_rate=0.01,momentum=0.1)

    run_all(learning_rate=0.03,momentum=0.9)
    run_all(learning_rate=0.03,momentum=0.6)
    run_all(learning_rate=0.03,momentum=0.3)
    run_all(learning_rate=0.03,momentum=0.1)

    run_all(learning_rate=0.05,momentum=0.9)
    run_all(learning_rate=0.05,momentum=0.6)
    run_all(learning_rate=0.05,momentum=0.3)
    run_all(learning_rate=0.05,momentum=0.1)

    run_all(learning_rate=0.07,momentum=0.9)
    run_all(learning_rate=0.07,momentum=0.6)
    run_all(learning_rate=0.07,momentum=0.3)
    run_all(learning_rate=0.07,momentum=0.1)

    run_all(learning_rate=0.09,momentum=0.9)
    run_all(learning_rate=0.09,momentum=0.6)
    run_all(learning_rate=0.09,momentum=0.3)
    run_all(learning_rate=0.09,momentum=0.1)

    run_all(learning_rate=0.1,momentum=0.9)
    run_all(learning_rate=0.1,momentum=0.6)
    run_all(learning_rate=0.1,momentum=0.3)
    run_all(learning_rate=0.1,momentum=0.1)

    run_all(learning_rate=0.3,momentum=0.9)
    run_all(learning_rate=0.3,momentum=0.6)
    run_all(learning_rate=0.3,momentum=0.3)
    run_all(learning_rate=0.3,momentum=0.1)

    run_all(learning_rate=0.5,momentum=0.9)
    run_all(learning_rate=0.5,momentum=0.6)
    run_all(learning_rate=0.5,momentum=0.3)
    run_all(learning_rate=0.5,momentum=0.1)

    run_all(learning_rate=0.9,momentum=0.9)
    run_all(learning_rate=0.9,momentum=0.6)
    run_all(learning_rate=0.9,momentum=0.3)
    run_all(learning_rate=0.9,momentum=0.1)
if __name__ == '__main__':
    main()