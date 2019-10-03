# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %%
from IPython.core.ultratb import AutoFormattedTB
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
import torch
from IPython import get_ipython

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

__ITB__ = AutoFormattedTB(mode='Verbose', color_scheme='LightBg', tb_offset=1)

assert torch.cuda.is_available(), "You need to request a GPU from Runtime > Change Runtime"


# %%
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


# class CrossEntropyLoss(nn.Module):
#     def __init__(self, weight=None, size_average=None, ignore_index=-100,
#                  reduce=None, reduction='mean'):
#         self.__dict__.update(locals())
#         super(CrossEntropyLoss, self).__init__()

# %%
# You'll probably want a function or something to test input / output sizes of the ConvTranspose2d layer

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


# %%
# Since you will be using the output of one network in two places(convolution and maxpooling),
# you can't use nn.Sequential.
# Instead you will write up the network like normal variable assignment as the example shown below:
# You are welcome (and encouraged) to use the built-in batch normalization and dropout layer.

# TODO: You need to change this to fit the UNet structure!!!
class CancerDetection(nn.Module):
    def __init__(self):
        super(CancerDetection, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

    def forward(self, input):
        conv1_out = self.conv1(input)
        relu2_out = self.relu2(conv1_out)
        conv3_out = self.conv3(relu2_out)
        relu4_out = self.relu4(conv3_out)
        return relu4_out


# 5 blocks
class ConvBlock(nn.Module):
    def __init__(self, c, h, w, num_layers=3):
        for i in range(num_layers):
            # create some Conv2d layers
            # use nn.Sequential
            pass



class TwoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoConv2d, self).__init__()

        self.conv2d = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3,3), padding=(1,1)),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels, out_channels, (3,3), padding=(1,1)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(out_channels)
                                    )

    def forward(self, x):
        return self.conv2d(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, (3,3), padding=(1,1)),
                                      nn.ReLU()
                                      )

    def forward(self, x):
        return self.upsample(x)

class DeepUNetwork(nn.Module):
    def __init__(self, dataset, initialization_strategy='xav'):
        super(DeepUNetwork, self).__init__()
        x, y = dataset[0]
        c, h, w = x.size()  # c=numchannels, h=height, w=width
        out = 2  # number of final output channels; correspond to number of classes

        # do conv transpose; instead of input to output shape size, it's output to input

        # Initialize like this but using ConvBlock
        self.conv0 = TwoConv2d(c,      64)
        self.conv1 = TwoConv2d(64,    128)
        self.conv2 = TwoConv2d(128,   256)
        self.conv3 = TwoConv2d(256,   512)
        self.conv4 = TwoConv2d(512,  1024)
        self.conv5 = TwoConv2d(1024,  512)
        self.conv6 = TwoConv2d(512,   256)
        self.conv7 = TwoConv2d(256,   128)
        self.conv8 = TwoConv2d(128,    64)
        self.conv9 = nn.Conv2d(64,    out, (1, 1), padding=(0, 0))

        self.transpose_conv0 = Upsample(1024, 512)
        self.transpose_conv1 = Upsample( 512, 256)
        self.transpose_conv2 = Upsample( 256, 128)
        self.transpose_conv3 = Upsample( 128,  64)

        self.max_pool = nn.MaxPool2d(2)
        
        # output is 512x512  (what help sesh video says)
        # padding of 1 keeps numbers more consistent
        # 3x3x64 filter, 64 unique kernels at beginning, means 64 features from that layer
        # upscaling - matrix multiplacation, do transpose
        # accuracy for img sgmntation: intersection over union (IoU metric)
        # do Sigmoid at end to get probabilities, then compute_accuracy to get 0s or 1s

    def forward(self, x):
        # n, c, h, w = x.size()
        gc.collect()
        # torch.cuda.empty_cache()

        # torch.cat((o0, o0), 1)  # concat convolvs along channel axis
        layer0 = self.conv0(x)  # Output is 64 x 512 x 512
        layer1 = self.conv1(self.max_pool(layer0))
        layer2 = self.conv2(self.max_pool(layer7))
        layer7 = self.conv7(torch.cat((layer2, self.transpose_conv2(layer2)), 1))
        layer8 = self.conv8(torch.cat((layer1, self.transpose_conv3(layer7)), 1))
        layer8 = self.conv3(self.max_pool(layer9))
        layer9 = self.conv9(layer8)

        return layer9

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
        return len(self.data)  # 100


# %%
# Create your datasets and neural network as you have before

########################### CREATE DATASETS AND NN #####################################

try:
    # your code for calling dataset and dataloader
    train_dataset = FashionMNISTProcessedDataset(
        '/home/seth/Downloads/fasionmnist', train=True)
    val_dataset = FashionMNISTProcessedDataset(
        '/home/seth/Downloads/fasionmnist', train=False)
    model = DeepUNetwork(train_dataset).cuda()
    
    # Compute and print the number of parameters in the model
    print('number of trainable parameters =', count_parameters(model))
    
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 8
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            shuffle=True, 
                            batch_size=batch_size,
                            num_workers=8,
                            pin_memory=True)

except:
    __ITB__()

# %%

def compute_accuracy(y_hat, y_truth):
    # 1 x (2 x 512 x 512)
    # 1 x (1 x 512 x 512)
    y_hat = y_hat[0]
    y_truth = y_truth[0]
    y_hat[0] = y_hat[0] > 0.5  # first channel image
    y_hat[1] = y_hat[1] < 0.5  # second channel image
    intersection = torch.sum(y_hat[0] * y_truth)
    union = torch.sum(y_hat[0] + y_truth >= 1)
    return intersection/union

num_epochs = 5
t_len = len(train_loader)
v_len = len(val_loader)
loops_per_epoch = t_len + (t_len // val_interval + 1) * (v_len-1)

train_loss_hist = []
train_accuracy_hist = []
val_loss_hist = []
val_accuracy_hist = []

val_losses = []
val_accuracies = []

val_interval = 50  # num of iterations between validation tests
val_loss_mean = 0
val_accuracy_mean = 0

gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated(0) / 1e9)

time_start = time.time()
try:
    # Call your model, figure out loss and accuracy
    for epoch in range(num_epochs):

        loop = tqdm(total=loops_per_epoch, position=0, leave=False)

        for loop_count, (x, y_truth) in enumerate(train_loader):
            x, y_truth = x.cuda(async=True), y_truth.cuda(async=True)

            optimizer.zero_grad()
            y_hat = model(x)
            
            # Accuracy calculation
            train_accuracy = compute_accuracy(y_hat, y)
            train_accuracy_hist.append(train_accuracy)

            # Add to loss and scores histories
            loss = objective(y_hat, y_truth.long)
            train_loss_hist.append(loss.item())

            # Update tqdm display
            print(torch.cuda.memory_allocated(0) / 1e9)
            loop.set_description(
                f'epoch:{epoch}, \
                loss:{loss.item():.4f}, \
                train_accuracy:{train_accuracy:.3f}, \
                mem:{mem:.2f}')
            loop.update()

            loss.backward()
            optimizer.step()


            # Validation section
            if loop_count != 0 and loop_count % val_interval == 0:

                for i, (x_val, y_truth_val) in enumerate(val_loader):

                    x_val, y_truth_val = x_val.cuda(async=True),  \
                                         y_truth_val.cuda(async=True)
                    y_hat_val = model(x_val)
                    val_losses.append(objective(y_hat_val, y_truth_val).item())
                    val_accuracies.append(compute_accuracy(y_hat_val, y_truth_val))
                    loop.update()

                val_loss_mean = np.mean(val_losses)
                val_accuracy_mean = np.mean(val_accuracies)

            val_loss_hist.append(val_loss_mean)
            val_accuracy_hist.append(val_score_mean)

    loop.update()
    elapsed_time = time.time() - time_start
    loop.update()
    loop.close()

except:
    __ITB__()

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
import seaborn as sns
sns.set_style('whitegrid')
custom = sns.color_palette("Paired", 9)
sns.set_palette(custom)
sns.set()

print(f'\n   --- Total time elapsed: {elapsed_time:.2f} ---\n')

plt.rcParams['figure.figsize'] = [35, 10]
fig, axs = plt.subplots(1, 2)

axs[0].plot(train_loss_hist, label='train')
axs[0].plot(val_loss_hist, label='val')
axs[0].legend()
axs[0].set_title('Loss')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

axs[1].plot(train_accuracy_hist, label='train')
axs[1].plot(val_accuracy_hist, label='val')
axs[1].legend()
axs[1].set_title('Accuracy')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Percent accuracy')
axs[1].grid(True)


plt.show()

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
