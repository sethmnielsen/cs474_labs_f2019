# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
# <a href="https://colab.research.google.com/github/sethmnielsen/cs474_labs_f2019/blob/master/DL_Lab3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#%% [markdown]
# # Lab 3: Intro to CNNs and DNNs
# 
# ## Objectives
# 
# * Build and train a deep conv net
# * Explore and implement various initialization techniques
# * Implement a parameterized module in Pytorch
# * Use a principled loss function
# 
# ## Video Tutorial
# [https://youtu.be/3TAuTcx-VCc](https://youtu.be/3TAuTcx-VCc)
# 
# ## Deliverable
# For this lab, you will submit an ipython notebook via learningsuite.
# This is where you build your first deep neural network!
# 
# For this lab, we'll be combining several different concepts that we've covered during class,
# including new layer types, initialization strategies, and an understanding of convolutions.
# 
# ## Grading Standards:
# * 30% Part 0: Successfully followed lab video and typed in code
# * 20% Part 1: Re-implement Conv2D and CrossEntropy loss function
# * 20% Part 2: Implement different initialization strategies
# * 10% Part 3: Print parameters, plot train/test accuracy
# * 10% Part 4: Convolution parameters quiz
# * 10% Tidy and legible figures, including labeled axes where appropriate
# ___
# 
# ### Part 0
# Watch and follow video tutorial:
# 
# [https://youtu.be/3TAuTcx-VCc](https://youtu.be/3TAuTcx-VCc)
# 
# **TODO:**
# 
# * Watch tutorial
# 
# **DONE:**

#%%
get_ipython().system('pip3 install torch')
get_ipython().system('pip3 install torchvision')
get_ipython().system('pip3 install tqdm')

#%% [markdown]
# 

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter
import pdb

assert torch.cuda.is_available(), "You need to request a GPU from Runtime > Change Runtime"


#%%
# Use the dataset class you created in lab2
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

#%% [markdown]
# ___
# 
# ### Part 1
# Re-implement a Conv2D module with parameters and a CrossEntropy loss function.
# 
# **TODO:**
# 
# * CrossEntropyLoss 
# * Conv2D
# 
# **DONE:**
# 
# ___
# 
# ### Part 2
# Implement a few initialization strategies which can include Xe initialization
# (sometimes called Xavier), Orthogonal initialization, and uniform random.
# You can specify which strategy you want to use with a parameter. 
# 
# 
# 
# Helpful links include:
# *  [Orthogonal Initialization](https://hjweide.github.io/orthogonal-initialization-in-convolutional-layers) (or the original paper: http://arxiv.org/abs/1312.6120)
# *  http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
# 
# **TODO:**
# * Parameterize custom Conv2D for different initilization strategies
# * Xe
# * Orthogonal
# * Uniform
# 
# **DONE:**
# 
# 

#%%
class CrossEntropyLoss(nn.Module):
  pass

class Conv2d(nn.Module):
  pass


#%%
class ConvNetwork(nn.Module):
  pass


#%%
# Initialize Datasets

# Initialize DataLoaders

# Initialize Model

# Initialize Objective and Optimizer and other parameters


#%%

# Run your training and validation loop and collect stats

#%% [markdown]
# 
# ___
# 
# ### Part 3
# Print the number of parameters in your network and plot accuracy of your training and validation 
# set over time. You should experiment with some deep networks and see if you can get a network 
# with close to 1,000,000 parameters.
# 
# **TODO:**
# * Experiment with Deep Networks
# * Plot accuracy of training and validation set over time
# * Print out number of parameters in the model 
# 
# **DONE:**
# 

#%%

# Go back up and try a few different networks and initialization strategies
# Plot loss if you want
# Plot accuracy


#%%
# Compute and print the number of parameters in the model

#%% [markdown]
# ___
# 
# ### Part 4
# Learn about how convolution layers affect the shape of outputs, and answer the following quiz questions. Include these in a new markdown cell in your jupyter notebook.
# 
# 
# *Using a Kernel size of 3×3 what should the settings of your 2d convolution be that results in the following mappings (first answer given to you)*
# 
# * (c=3, h=10, w=10) ⇒ (c=10, h=8, w=8) : (out_channels=10, kernel_size=(3, 3), padding=(0, 0))
# * (c=3, h=10, w=10) ⇒ (c=22, h=10, w=10) : **Your answer in bold here**
# * (c=3, h=10, w=10) ⇒ (c=65, h=12, w=12) : **Your answer in bold here**
# * (c=3, h=10, w=10) ⇒ (c=7, h=20, w=20) : **Your answer in bold here**
# 
# *Using a Kernel size of 5×5:*)
# 
# * (c=3, h=10, w=10) ⇒ (c=10, h=8, w=8) : (out_channels=10, kernel_size=(5, 5), padding=(1, 1))
# * (c=3, h=10, w=10) ⇒ (c=100, h=10, w=10) : **Your answer in bold here**
# * (c=3, h=10, w=10) ⇒ (c=23, h=12, w=12) : **Your answer in bold here**
# * (c=3, h=10, w=10) ⇒ (c=5, h=24, w=24) : **Your answer in bold here**
# 
# *Using Kernel size of 5×3:*
# 
# * (c=3, h=10, w=10) ⇒ (c=10, h=8, w=8) : **Your answer in bold here**
# * (c=3, h=10, w=10) ⇒ (c=100, h=10, w=10) : **Your answer in bold here**
# * (c=3, h=10, w=10) ⇒ (c=23, h=12, w=12) : **Your answer in bold here**
# * (c=3, h=10, w=10) ⇒ (c=5, h=24, w=24) : **Your answer in bold here**
# 
# *Determine the kernel that requires the smallest padding size to make the following mappings possible:*
# 
# * (c=3, h=10, w=10) ⇒ (c=10, h=9, w=7) : **Your answer in bold here**
# * (c=3, h=10, w=10) ⇒ (c=22, h=10, w=10) : **Your answer in bold here**
# 
# **TODO:**
# 
# * Answer all the questions above 
# 
# **DONE:**
# 

#%%
# Write some test code for checking the answers for these problems (example shown in the video)


