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
    def __init__(self, dataset, in_dim=784, out_dim=10):
        super(LinearNetwork,self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim))

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
        return len(self.data)

train_dataset = FashionMNISTProcessedDataset('/tmp/fashionmnist', train=True)
xt, yt = train_dataset[0]
in_dim = xt.size(0)
out_dim = yt
model = LinearNetwork(train_dataset, 784, 10)
model = model.cuda()
objective = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)
train_loader = DataLoader(train_dataset,
                          batch_size=42,
                          pin_memory=True)

for x, y_truth in train_loader:
    x, y_truth = x.cuda(async=True), y_truth.cuda(async=True)
    
    optimizer.zero_grad()
     
    y_hat = model(x)
    loss = objective(y_hat, y_truth)
    
    loss.backward()
    
    print(loss)
    break

weight, bias = list(model.parameters())
# Create a dataset class that extends the torch.utils.data Dataset class here

# Extend the torch.Module class to create your own neural network

# Instantiate the train and validation sets

# Instantiate your data loaders

# Instantiate your model and loss and optimizer functions

# Run your training / validation loops
