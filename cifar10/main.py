import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) #output layer
        
        return x

def main():
    data_dir = './dataset'
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=2)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = Net()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epoch_losses = []
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
            if i % 2000 == 1999:
                print('%d, %5d| loss: %.3f' %(epoch+1, i+1, running_loss/2000))
                saved_loss = running_loss
                running_loss = 0.0
        epoch_losses.append(saved_loss/10000)
    print('Training done!')
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: %d %%' % (100 * correct / total))
"""
    epochs = range(1,21)
    plt.plot(epochs, epoch_losses, 'g', label='Training loss')
    plt.title('Trainingloss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()    
"""       
"""
    print(len(trainset))
    sample = next(iter(trainset))
    print(len(sample))
    image, label = sample
    print(type(image))
    print(type(label))
    image.shape
    print(len(testset))

    # train_loader images
    dataiter = iter(train_loader)
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(classes[labels[i]])
        image = images[i].numpy()
        plt.imshow(np.rot90(image.T, k=3))
        plt.show()

    # test_loader images
    dataiter = iter(test_loader)
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(classes[labels[i]])
        image = images[i].numpy()
        plt.imshow(np.rot90(image.T, k=3))
        plt.show()
"""
if __name__ == '__main__':
    main()
