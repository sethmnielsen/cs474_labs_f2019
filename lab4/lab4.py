# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

# %% [markdown]
# <a href="https://colab.research.google.com/github/sethmnielsen/cs474_labs_f2019/blob/master/DL_Lab4.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Lab 4: Cancer Detection
#
# ## Objective
# * To build a dense prediction model
# * To begin reading current papers in DNN research
#
# ## Deliverable
# For this lab, you will turn in a notebook that describes your efforts at creating
# a pytorch radiologist. Your final deliverable is a notebook that has (1) deep network,
# (2) cost function, (3) method of calculating accuracy,
# (4) an image that shows the dense prediction produced by your network on the pos_test_000072.png image.
# This is an image in the test set that your network will not have seen before.
# This image, and the ground truth labeling, is shown below.
# (And is contained in the downloadable dataset below).
#
# ![](http://liftothers.org/dokuwiki/lib/exe/fetch.php?w=200&tok=a8ac31&media=cs501r_f2016:pos_test_000072_output.png)
# <img src="http://liftothers.org/dokuwiki/lib/exe/fetch.php?media=cs501r_f2016:pos_test_000072.png" width="200">
#
#
# ## Grading standards
# Your notebook will be graded on the following:
# * 40% Proper design, creation and debugging of a dense prediction network
# * 40% Proper implementation of a loss function and train/test set accuracy measure
# * 10% Tidy visualizations of loss of your dense predictor during training
# * 10% Test image output
#
#
# ## Data set
# The data is given as a set of 1024×1024 PNG images. Each input image (in
# the ```inputs``` directory) is an RGB image of a section of tissue,
# and there a file with the same name (in the ```outputs``` directory)
# that has a dense labeling of whether or not a section of tissue is cancerous
# (white pixels mean “cancerous”, while black pixels mean “not cancerous”).
#
# The data has been pre-split for you into test and training splits.
# Filenames also reflect whether or not the image has any cancer at all
# (files starting with ```pos_``` have some cancerous pixels, while files
# starting with ```neg_``` have no cancer anywhere).
# All of the data is hand-labeled, so the dataset is not very large.
# That means that overfitting is a real possibility.
#
# ## Description
# For a video including some tips and tricks that can help with this lab: [https://youtu.be/Ms19kgK_D8w](https://youtu.be/Ms19kgK_D8w)
# For this lab, you will implement a virtual radiologist.
# You are given images of possibly cancerous tissue samples,
# and you must build a detector that identifies where in the tissue cancer may reside.
#
# ---
#
# ### Part 0
# Watch and follow video tutorial:
#
# https://youtu.be/Ms19kgK_D8w
#
# **TODO:**
#
# * Watch tutorial
#
# **DONE:**
#
# ### Part 1
# Implement a dense predictor
#
# In previous labs and lectures, we have talked about DNNs that classify an
# entire image as a single class. Here, however, we are interested in a more nuanced classification:
# given an input image, we would like to identify each pixel that is possibly cancerous.
# That means that instead of a single output, your network should output an “image”,
# where each output pixel of your network represents the probability that a pixel is cancerous.
#
# **TODO:**
#
# * Create a Network that classifies each pixel as a 1 or 0 for cancerous / not cancerous
#
# **DONE:**
#
# ___
#
# ### Part 1a
# Implement your network topology
#
#
# Use the “Deep Convolution U-Net” from this paper: [(U-Net: Convolutional Networks for Biomedical Image Segmentation)](https://arxiv.org/pdf/1505.04597.pdf)
#
# ![(Figure 1)](https://lh3.googleusercontent.com/qnHiB3B2KRxC3NjiSDtY08_DgDGTDsHcO6PP53oNRuct-p2QXCR-gyLkDveO850F2tTAhIOPC5Ha06NP9xq1JPsVAHlQ5UXA5V-9zkUrJHGhP_MNHFoRGnjBz1vn1p8P2rMWhlAb6HQ=w2400)
#
# You should use existing pytorch functions (not your own Conv2D module), such as ```nn.Conv2d```;
# you will also need the pytorch function ```torch.cat``` and ```nn.ConvTranspose2d```
#
# ```torch.cat``` allows you to concatenate tensors.
# ```nn.ConvTranspose2d``` is the opposite of ```nn.Conv2d```.
# It is used to bring an image from low res to higher res.
# [This blog](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0) should help you understand this function in detail.
#
# Note that the simplest network you could implement (with all the desired properties)
# is just a single convolution layer with two filters and no relu!
# Why is that? (of course it wouldn't work very well!)
#
# **TODO:**
#
# * Understand the U-Net architecture
# * Understand concatenation of inputs from multiple prior layers
# * Understand ConvTranspose
# * Answer Question / Reflect on simplest network with the desired properties
#
# **DONE:**
#
#
# ___
# The intention of this lab is to learn how to make deep neural nets and implement loss function.
# Therefore we'll help you with the implementation of Dataset.
# This code will download the dataset for you so that you are ready to use it and focus on network
# implementation, train_loss_hist and accuracies.

# %%
# get_ipython().system('pip3 install torch')
# get_ipython().system('pip3 install torchvision')
# get_ipython().system('pip3 install tqdm')

from IPython.core.ultratb import AutoFormattedTB
import sys
import traceback
import gc
import tarfile
import gzip
import os
import torchvision
import pdb
from torch.nn.parameter import Parameter
from tqdm import tqdm
from torchvision import transforms, utils, datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import torch
import time
from IPython import get_ipython
from typing import *

# Live plotting stuff
import seaborn as sns
sns.set_style('whitegrid')
custom = sns.color_palette("Paired", 9)
sns.set_palette(custom)
sns.set()
plt.ion()

__ITB__ = AutoFormattedTB(mode='Verbose', color_scheme='LightBg', tb_offset=1)

assert torch.cuda.is_available(), "You need to request a GPU from Runtime > Change Runtime"


# %%
"""
    ########################### DOWNLOAD DATASET #####################################
"""
class CancerDataset(Dataset):
    def __init__(self, root, download=True, size=512, train=True):
        if download and not os.path.exists(os.path.join(root, 'cancer_data')):
            datasets.utils.download_url(
                'http://liftothers.org/cancer_data.tar.gz', root, 'cancer_data.tar.gz', None)
            self.extract_gzip(os.path.join(root, 'cancer_data.tar.gz'))
            self.extract_tar(os.path.join(root, 'cancer_data.tar'))

        postfix = 'train' if train else 'test'
        root = os.path.join(root, 'cancer_data', 'cancer_data')
        self.dataset_folder = torchvision.datasets.ImageFolder(os.path.join(
            root, 'inputs_' + postfix), transform=transforms.Compose([transforms.Resize(size), transforms.ToTensor()]))
        self.label_folder = torchvision.datasets.ImageFolder(os.path.join(
            root, 'outputs_' + postfix), transform=transforms.Compose([transforms.Resize(size), transforms.ToTensor()]))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    @staticmethod
    def extract_tar(tar_path):
        print('Untarring {}'.format(tar_path))
        z = tarfile.TarFile(tar_path)
        z.extractall(tar_path.replace('.tar', ''))

    def __getitem__(self, index):
        img = self.dataset_folder[index]
        label = self.label_folder[index]
        return img[0] * 255, label[0][0]

    def __len__(self):
        return len(self.dataset_folder)

class DeepPlotter(object):
    """
    Demonstrates a basic example of the "scaffolding" you need to efficiently
    blit drawable artists on top of a background.
    """
    def __init__(self):
        
        self.fig, self.ax1, self.ax2, self.lines = self.setup_axes()

    def setup_axes(self):
        """Setup the figure/axes and plot any background artists."""
        fig = plt.figure(1)
        axs = fig.add_subplot(1, 1, 1)
        fig.clf()
        axs = plt.subplots(1, 2)
        ax1 : plt.axis = axs[0]
        ax2 : plt.axis = axs[1]
        fig.canvas.draw()
        
        line1_t, = ax1.plot([], label='train')
        line1_v, = ax1.plot([], label='val')

        ax1.set_title('Loss vs Iterations')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.autoscale()
        # ax1.legend()

        line2_t, = ax2.plot([], label='train')
        line2_v, = ax2.plot([], label='val')

        ax2.set_title('Accuracy vs Iterations')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Percent Accuracy')
        ax2.grid(True)
        ax2.autoscale()
        # ax2.legend()

        lines = [line1_t, line1_v, line2_t, line2_v]

        return fig, ax1, ax2, lines

    def update(self, xdata=[], ydata=[]):
        """Update the artist for any changes to self.xy."""
        for i in range(4):
            self.lines[i].set_data(xdata[i], ydata[i])

        self.blit()

    def blit(self):
        """
        Efficiently update the figure, without needing to redraw the
        "background" artists.
        """
        # self.ax1.draw_artist(self.lines[:2])
        # self.ax2.draw_artist(self.lines[2:4])
        self.ax1.autoscale()
        self.ax2.autoscale()
        self.ax1.redraw_in_frame()
        self.ax2.redraw_in_frame()
        self.fig.canvas.blit(self.fig.bbox)

# %% [markdown]
# ___
#
# ### Part 1b
# Implement a cost function
#
# You should still use cross-entropy as your cost function, but you may need to think hard about how exactly to set this up – your network should output cancer/not-cancer probabilities for each pixel, which can be viewed as a two-class classification problem.
#
# **TODO:**
#
# * Adapt CrossEntropyLoss for 2 class pixel classification
#
# **DONE:**
#
#

# %%
"""
    ########################### DEFINITIONS #####################################
"""
# Since you will be using the output of one network in two places(convolution and maxpooling),
# you can't use nn.Sequential.
# Instead you will write up the network like normal variable assignment as the example shown below:
# You are welcome (and encouraged) to use the built-in batch normalization and dropout layer.

class TwoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoConv2d, self).__init__()

        self.conv2d = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3,3), padding=(1,1)),
                                    nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels, (3,3), padding=(1,1)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(out_channels) )

    def forward(self, x):
        return self.conv2d(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (4, 4), stride=2, padding=(1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.upsample(x)


class DeepUNetwork(nn.Module):
    def __init__(self, initialization_strategy='xav'):
        super(DeepUNetwork, self).__init__()
        out = 2  # number of final output channels; correspond to number of classes

        # do conv transpose; instead of input to output shape size, it's output to input

        # Initialize like this but using ConvBlock
        self.conv1 = TwoConv2d(3,      64)
        self.conv2 = TwoConv2d(64,    128)
        self.conv3 = TwoConv2d(128,   256)
        self.conv4 = TwoConv2d(256,   512)  

        self.conv5 = TwoConv2d(512,  1024)
        
        # upsample
        self.conv6 = TwoConv2d(1024,  512)
        # upsample
        self.conv7 = TwoConv2d(512,   256)
        # upsample
        self.conv8 = TwoConv2d(256,   128)
        # upsample
        self.conv9 = TwoConv2d(128,    64)
        self.conv_out = nn.Conv2d(64,    out, (1, 1), padding=(0, 0)) # output segmented map

        self.up_conv6 = Upsample(1024, 512)
        self.up_conv7 = Upsample( 512, 256)
        self.up_conv8 = Upsample( 256, 128)
        self.up_conv9 = Upsample( 128,  64)

        self.max_pool = nn.MaxPool2d(2)
        
        # output is 512x512  (what help sesh video says)
        # padding of 1 keeps numbers more consistent
        # 3x3x64 filter, 64 unique kernels at beginning, means 64 features from that layer
        # upscaling - matrix multiplacation, do transpose
        # accuracy for img sgmntation: intersection over union (IoU metric)
        # do Sigmoid at end to get probabilities, then compute_accuracy to get 0s or 1s

    def forward(self, x):
        layer1 = self.conv1(x)  # Output is 64 x 512 x 512
        layer2 = self.conv2(self.max_pool(layer1))
        layer3 = self.conv3(self.max_pool(layer2))
        layer4 = self.conv4(self.max_pool(layer3))
        
        layer5 = self.conv5(self.max_pool(layer4))

        layer6 = self.conv6( torch.cat(( layer4, self.up_conv6(layer5) ), dim=1) )
        layer7 = self.conv7( torch.cat(( layer3, self.up_conv7(layer6) ), dim=1) )
        layer8 = self.conv8( torch.cat(( layer2, self.up_conv8(layer7) ), dim=1) )
        layer9 = self.conv9( torch.cat(( layer1, self.up_conv9(layer8) ), dim=1) )

        output_layer: Tensor = self.conv_out(layer9)
        
        return output_layer


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x)
                                          for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param


def compute_accuracy(y_hat: Tensor, y_truth: Tensor):
    # 8 x (2 x 512 x 512)  y_hat
    # 8 x (    512 x 512)  y_truth
    y_hat_hot = torch.argmax(y_hat, dim=1)
    result = y_hat_hot.byte() ^ y_truth.byte()
    accuracy = 1 - torch.sum( result ).item() / result.numel() 
    return accuracy


# %%
"""
    ########################### INITIALIZATION #####################################
"""
# Create your datasets and neural network as you have before

try:
    # your code for calling dataset and dataloader
    train_dataset = CancerDataset(
        '/home/seth/Downloads', download=False, train=True, size=256)
    val_dataset = CancerDataset(
        '/home/seth/Downloads', download=False, train=False, size=256)
    
    model = DeepUNetwork(train_dataset).cuda(0)
    
    # Compute and print the number of parameters in the model
    # print('number of trainable parameters =', count_parameters(model))
    
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 12
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=4,
                            pin_memory=True)

                            

    
except Exception as e:
    # __ITB__()
    print(traceback.format_exc())
    sys.exit(e)

#%%
"""
    ########################### START MAIN #####################################
"""

try:
    train_loss_hist: List[float] = []
    train_accuracy_hist: List[float] = []
    val_loss_x: List[float] = []
    val_loss_y: List[float] = []
    val_accuracy_x: List[float] = []
    val_accuracy_y: List[float] = []

    val_losses: List[float] = []
    val_accuracies: List[float] = []

    val_interval = 50  # num of iterations between validation tests
    val_loss_mean = 0
    val_accuracy_mean = 0

    num_epochs = 6
    iteration_count = 0
    t_len = len(train_loader)
    v_len = len(val_loader)
    loops_per_epoch = t_len + (t_len // val_interval + 1) * v_len

    # dp = DeepPlotter()
    # # time_arr = np.arange(loops_per_epoch*num_epochs)
    # time_arr = []
    # xdata = [time_arr, val_loss_x, time_arr, val_accuracy_x]
    # ydata = [train_loss_hist, val_loss_y, train_accuracy_hist, val_accuracy_y]
    
    time_start = time.time()
    # Call your model, figure out loss and accuracy
    for epoch in range(num_epochs):

        loop = tqdm(total=loops_per_epoch, position=0, leave=False)

        for batch_num, (x, y_truth) in enumerate(train_loader):
            x, y_truth = x.cuda(async=True), y_truth.cuda(async=True)

            optimizer.zero_grad()
            y_hat = model(x)
            
            # Accuracy calculation
            train_accuracy = compute_accuracy(y_hat, y_truth)

            # Add to loss and scores histories
            loss = objective(y_hat, y_truth.long())

            # Update tqdm display
            loop.set_description(
                f'epoch:{epoch+1}, loss:{loss.item():.2f}, train_accuracy:{train_accuracy:.3f}')
            loop.update()

            loss.backward()
            optimizer.step()


            # Validation section
            if batch_num % val_interval == 0:

                with torch.no_grad():
                    val_losses = []
                    val_accuracies = []
                    for i, (x_val, y_truth_val) in enumerate(val_loader):

                        x_val, y_truth_val = x_val.cuda(async=True),  \
                                            y_truth_val.cuda(async=True)
                        y_hat_val = model(x_val)
                        val_losses.append(objective(y_hat_val, y_truth_val.long()).item())
                        val_accuracies.append(compute_accuracy(y_hat_val, y_truth_val))
                        loop.update()
                    
                    val_loss_mean = np.mean(val_losses)
                    val_accuracy_mean = np.mean(val_accuracies)

                    val_loss_x.append(iteration_count)
                    val_loss_y.append(val_loss_mean)
                    val_accuracy_x.append(iteration_count)
                    val_accuracy_y.append(val_accuracy_mean)

            # train_time_arr = time_arr[:iteration_count]
            # xdata = [train_time_arr, val_loss_x, train_time_arr, val_accuracy_x]
            # ydata = [train_loss_hist, val_loss_x, train_time_arr, val_accuracy_x]

            # Update plot 
            # time_arr.append(iteration_count)
            train_loss_hist.append(loss.item())
            train_accuracy_hist.append(train_accuracy)
            
            # dp.update(xdata, ydata)
            
            # end of loop
            iteration_count += 1

    elapsed_time = time.time() - time_start
    loop.update()
    loop.close()

except Exception as e:
    # __ITB__()
    print(traceback.format_exc())
    sys.exit(e)

# %% [markdown]

#
# ___
#
# ### Part 2
#
# Plot performance over time
#
# Please generate a plot that shows loss on the training set as a function of training time. Make sure your axes are labeled!
#
# **TODO:**
#
# * Plot training loss as function of training time (not Epochs)
#
# **DONE:**
#

# %%
# Your plotting code here
print(f'\n   --- Total time elapsed: \t{elapsed_time:.2f}\t---')
print(f'   --- Number of iterations: \t{iteration_count}\t---')
print(f'   --- Avg speed (it/s): \t{(iteration_count/elapsed_time):.2f}\t---')
print(f'   --- Avg duration (s/epoch): \t{(elapsed_time/num_epochs):.2f}\t---\n')

plt.rcParams['figure.figsize'] = [10, 4]
fig, axs = plt.subplots(1, 2)

axs[0].plot(train_loss_hist, label='train')
axs[0].plot(val_loss_x, val_loss_y, label='val')
axs[0].legend()
axs[0].set_title('Loss')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

axs[1].plot(train_accuracy_hist, label='train')
axs[1].plot(val_accuracy_x, val_accuracy_y, label='val')
axs[1].legend()
axs[1].set_title('Accuracy')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Percent accuracy')
axs[1].grid(True)

# %% [markdown]
# ___
#
# ### Part 3
#
# Generate a prediction on the pos_test_000072.png image
#
# Calculate the output of your trained network on the pos_test_000072.png image,
# then make a hard decision (cancerous/not-cancerous) for each pixel.
# The resulting image should be black-and-white, where white pixels represent things
# you think are probably cancerous.
#
# **TODO:**
#
# **DONE:**
#
# **NOTE:**
#
# Guessing that the pixel is not cancerous every single time will give you an accuracy of ~ 85%.
# Your trained network should be able to do better than that (but you will not be graded on accuracy).
# This is the result I got after 1 hour or training.
#
# ![](http://liftothers.org/dokuwiki/lib/exe/fetch.php?w=400&tok=d23e0b&media=cs501r_f2016:training_accuracy.png)
# ![](http://liftothers.org/dokuwiki/lib/exe/fetch.php?w=400&tok=bb8e3c&media=cs501r_f2016:training_loss.png)

# %%
# Code for testing prediction on an image
test_img, label = val_dataset[172]
y_test = model( test_img.cuda(async=True).unsqueeze(0) )
y_test = y_test.detach().squeeze().cpu().numpy()
y_test_hot = torch.argmax(Tensor(y_test), dim=0)
result = y_test_hot.byte() ^ label.byte()
accuracy_final = 1 - torch.sum( result ).item() / result.numel() 

print(f'\n   --- Score for test image: {accuracy_final:.3f} ---\n')

# Plot the result
plt.rcParams["axes.grid"] = False
fig2 = plt.figure(2)
ax_im = fig2.add_subplot(1, 1, 1)
ax_im.imshow(y_test_hot, interpolation='nearest', cmap='gray')
ax_im.set_title('Cancer Detection - Test Result')

plt.show()


#%%
