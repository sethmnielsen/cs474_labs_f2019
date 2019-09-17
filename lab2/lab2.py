import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from tqdm import tqdm
 
assert torch.cuda.is_available() # You need to request a GPU from Runtime > Change Runtime Type

# Write the boilerplate code from the video here

class LinearNetwork(nn.Module):
    def __init__(self, in_dim=784, out_dim=10):
        super(LinearNetwork,self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 1000),
                                 nn.Sigmoid(),
                                 nn.Linear(1000, out_dim))

    def forward(self, x):
        n, c, h, w = x.size()
        flattened = x.view(n, c*h*w)
        return self.net(flattened)

class FashionMNISTProcessedDataset(Dataset):
    def __init__(self, root, train=True):
        self.data = datasets.FashionMNIST(root,
                                          train=train,
                                          transform=transforms.ToTensor(),
                                          download=True)
        
    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y

    def __len__(self):
        return 100 #len(self.data)

train_dataset = FashionMNISTProcessedDataset('/tmp/fashionmnist', train=True)
validation_dataset = FashionMNISTProcessedDataset('/tmp/fashionmnist', train=False)
model = LinearNetwork(784, 10)
model = model.cuda()
objective = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)
train_loader = DataLoader(train_dataset,
                          batch_size=10,
                          pin_memory=True)
validation_loader = DataLoader(validation_dataset,
                               batch_size=10,
                               pin_memory=True)

losses = np.array([])
train_losses = np.array([])
val_losses = np.array([])

epochs = 1000
loop = tqdm(total=len(train_loader)*2*epochs, position=0)

for epoch in range(epochs):
    for x, y_truth in train_loader:
        x, y_truth = x.cuda(async=True), y_truth.cuda(async=True)
        
        optimizer.zero_grad()
        
        y_hat = model(x)
        loss = objective(y_hat, y_truth)
        losses = np.append(losses,loss.item())
        loss.backward()

        optimizer.step()

        loop.update(1)
        
        del y_hat
        del loss

    # average loss for this epoch
    train_losses = np.append(train_losses, np.mean(losses))
    del losses
    losses = np.array([])

    with torch.no_grad():
        for x_val, y_val_truth in validation_loader:
            x_val, y_val_truth = x_val.cuda(async=True), y_val_truth.cuda(async=True)
            y_val_hat = model(x_val)
            loss = objective(y_val_hat, y_val_truth)
            losses = np.append(losses, loss.item())

            loop.update(1)

            del y_val_hat
            del loss

    val_losses = np.append(val_losses, np.mean(losses))
    del losses
    losses = np.array([])

    descrip = 'train loss:{:.4f}, val loss:{:.4f}'.format(train_losses[epoch], 
                                                          val_losses[epoch])
    loop.set_description(descrip)

loop.close()

# weight, bias = list(model.parameters())
# Create a dataset class that extends the torch.utils.data Dataset class here

# Extend the torch.Module class to create your own neural network

# Instantiate the train and validation sets

# Instantiate your data loaders

# Instantiate your model and loss and optimizer functions

# Run your training / validation loops
