# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a 
# href="https://colab.research.google.com/github/wingated/cs474_labs_f2019/blob/master/DL_Lab8.ipynb"
#   target="_parent">
#   <img
#     src="https://colab.research.google.com/assets/colab-badge.svg"
#     alt="Open In Colab"/>
# </a>
# %% [markdown]
# # Lab 8: GANs
# 
# ### Description
# In this lab, we will build our very first GAN. This can be frustrating at first, but the end result is really cool. We've tried to make the steps intuitive.
# 
# Here is what you will learn:
# * GANs are generative models that learn to generate data, based on a min-max/adversarial game between a Generator (G) and Discriminator (D).
# * The parameters of both Generator and Discriminator are optimized with Stochastic Gradient Descent (SGD) or RMSprop or Adam
# * How these concepts translate into pytorch code for GAN optimization.
# 
# Overview of the tutorial:
# 1. GAN intro
# 2. Defining the neural networks in pytorch, computing a forward pass
# 3. Training our GAN
# 
# This lab is modified from https://github.com/tomsercu/gan-tutorial-pytorch
# 
# ### Deliverable
# We have provided the GAN architecture for you. Your objective is to:
# 1. Create a DataLoader for the CelebA dataset.
# 2. Create a Dataset and a DataLoader for a dataset from a domain of your choice.
# 3. Implement the original GAN loss
# 4. Implement the training loop and train your GAN.
# 
# ### Grading Standards
# - 25% correctly load CelebA dataset and a dataset of your choice
# - 25% correctly implement the original GAN loss
# - 50% correctly implement the training loop and train your GAN (you will not be graded on quality of generated images)
# 
# ### Tips:
# - This lab is complex. Please read through the entire spec before diving in.
# - Also, note that training on this dataset will likely take some time. Please make sure you start early enough to run the training long enough!
# - Expected values: Discriminator Loss will hover around ~ 0.5, Generator Loss should hover around ~ 5.5. You should see discernible results within 1 epoch (~20-30 minutes of training on Colab).
# %% [markdown]
# # Some cool demos:
# * Progress over the last 5 years, from [Ian Goodfellow tweet](https://twitter.com/goodfellow_ian/status/1084973596236144640)
# 
# ![tweet image](https://github.com/tomsercu/gan-tutorial-pytorch/blob/master/figs/goodfellow_tweet.jpg?raw=1)
# 
# * CycleGAN translating horses into zebras: https://www.youtube.com/watch?v=9reHvktowLY
# * CycleGAN teaser: ![cyclegan teaser image](https://github.com/tomsercu/gan-tutorial-pytorch/blob/master/figs/cyclegan_teaser_high_res.jpg?raw=1)
# * High resolution faces with StyleGAN https://www.youtube.com/watch?v=kSLJriaOumA
# * https://ganbreeder.app web-interface to create images based on [BigGan](https://arxiv.org/abs/1809.11096)
# 
# %% [markdown]
# # 1. GAN first introduction
# [GAN picture](figs/gan_xavigiro.png)
# 
# <img src="https://github.com/tomsercu/gan-tutorial-pytorch/blob/master/figs/gan_xavigiro.png?raw=1" alt="GAN picture" style="width: 700px;"/>
# 
# GANs are a class of unsupervised generative models which implicitly model the data density.
# 
# The basic setup is pictured above. There are two "competing" neural networks:
# * The Generator wants to learn to generate realistic images that are indistinguishable from the real data. 
#     - *input*: Gaussian noise random sample. *output*: a (higher dimensional) datapoint
# * The Discriminator wants to tell the real & fake images apart.
#     - *input*: datapoint/image, *output*: probability assigned to datapoint being real. Think binary classifier.
# * The typical analogy: the generator is like a counterfeiter trying to look like real, the discriminator is the police trying to tell counterfeits from the real work.
# * The key novelty of GANs is to pass the error signal (gradients) from the discriminator to the generator: the generator neural network uses the information from the competing discriminator neural network to know how to produce more realistic output.
# %% [markdown]
# Let's start with defining the generator G and discriminator D in pytorch.
# %% [markdown]
# # 2. Define the Generator and Discriminator

# %%
# get_ipython().system('pip install torch==1.1.0')
# get_ipython().system('pip install torchvision==0.3.0')
import sys
print(sys.version) # python 3.6
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
print(torch.__version__) 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os, time

import itertools
import pickle
import imageio
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm

# You can use whatever display function you want. This is a really simple one that makes decent visualizations
def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1) # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())

# %% [markdown]
# ## Defining the neural networks

# %%
# helper function to initialize the weights using a normal distribution. 
# this was done in the original work (instead of xavier) and has been shown
# to help GAN performance
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# %%
class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x


# %%
class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x


# %%
#####
# instantiate a Generator and Discriminator according to their class definition.
#####
G = Generator()
D = Discriminator()
print(D)
print(G)

# %% [markdown]
# ## Testing the neural networks (forward pass)

# %%
samples = torch.randn(5, 3, 64,64) # batch size x channels x width x height
D(samples)

# %% [markdown]
# Things to try:
# * What happens if you change the number of samples in a batch?
# * What happens if you change the width/height of the input?
# * What are the weights of the discriminator? You can get an iterator over them with `.parameters()` and `.named_parameters()`

# %%
for name, p in D.named_parameters():
    print(name, p.shape)

# %% [markdown]
# We will think of the concatentation of all these discriminator weights in one big vector as $\theta_D$.
# 
# Similarly we name the concatentation of all the generator weights in one big vector $\theta_G$.

# %%
for name, p in G.named_parameters():
    print(name, p.shape)


# %%
# A small batch of 2 samples, random noise.
z = torch.randn(2, 100).view(-1,100,1,1)
x_gen = G(z)
#notice that the generated value is a batch of 2 images
x_gen.shape


# %%
z = torch.randn(8, 100).view(-1,100,1,1).cuda()
G = G.cuda()
show_imgs(G(z))

# %% [markdown]
# In traditional deep learning, you measure performance by looking at the loss value. In GANs, this does not work well because we are performing a Min-Max and the loss values may not be intuitively lower when the network is doing well. 
# 
# So, performance must be measured qualitatively, by looking at images. Therefore, you can sample random $z$ vectors every pass through the network to see how "novel" the generation is becoming. And you can also sample a single $z$ vector that is passed through the network every time to see how a single example progresses during training. 

# %%
fixed_z_ = torch.randn((5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), requires_grad=False)

# %% [markdown]
# ## Loading the data and computing forward pass

# %%
batch_size = 128
lr = 0.0002
train_epoch = 3

img_size = 64

#download the data, and change the filepath
data = datasets.CelebA('/tmp/CelebA', 'train', download=True)


# possible sources for celeba: torchvision datasets, or this google drive link: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
dataset = datasets.ImageFolder(root='/tmp/CelebA',
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
##### Create the dataloader #####
train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          pin_memory=True)

# %% [markdown]
# Dataset and DataLoader are abstractions to help us iterate over the data in random order.
# %% [markdown]
# Let's look at a sample:

# %%
ix=140
x, _ = dataset[ix]
show_imgs(x)

# %% [markdown]
# Feed the image into the discriminator; the output will be the probability the (untrained) discriminator assigns to this sample being real.

# %%
# for one image:
Dscore = D(x.unsqueeze(0))
Dscore


# %%
# How you can get a batch of images from the dataloader:
xbatch, _ = iter(train_loader).next()
print('xbatch shape:', xbatch.shape)
example = D(xbatch)
print('example from descriminator shape:', example.shape)
print(example)


# %%
show_imgs(xbatch)

# %% [markdown]
# # 3. Now to train your GAN
# %% [markdown]
# We introduced and defined the generator G, the discriminator D, and the dataloader which will give us minibatches of real data. 
# 
# To recap the basic idea of the min-max / adversarial game:
# * The Generator and Discriminator have competing objectives, they are "adversaries".
# * The Discriminator wants to assign high probability to real images and low probability to generated (fake) images
# * The Generator wants its generated images to look real, so wants to modify its outputs to get high scores from the Discriminator
# * We will optimize both alternatingly, with SGD steps (as before): optimize $\theta_D$ the weights of $D(x, \theta_D)$, and  $\theta_G$ the weights of $G(z, \theta_G)$.
# * Final goal of the whole min-max game is for the Generator to match the data distribution: $p_G(x) \approx p_{data}(x)$.
# 
# 
# Now what are the objective functions for each of them? As mentioned in the introduction, the objective for the discriminator is to classify the real images as real, so $D(x) = 1$, and the fake images as fake, so $D(G(z))=0$.
# This is a typical binary classification problem which calls for the binary cross-entropy (BCE) loss, which encourages exactly this solution.
# 
# For G we just try to minimize the same loss that D maximizes. See how G appears inside D? This shows how the output of the generator G is passed into the Discriminator to compute the loss.
# 
# 
# This is the optimization problem:
# 
# $$
# \min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
# $$
# 
# We will do a single SGD step alternatingly to maximize D, then minimize G.
# In fact for G we use a modified (non-saturing) loss $-\log D(G(z))$. Different modifications of the loss and the relation to the distance between distributions $p_{data}$ and $p_{G}$ became a topic of research over the last years.
# 
# BCE takes care of the log, you won't manually compute any Log values
# 

# %%
fixed_z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), requires_grad=False)

train_epoch = 3


G = Generator(128)
D = Discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G = G.cuda()
D = D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


# %%

def train_gan(G, D, G_optimizer, D_optimizer, BCE_loss, train_loader, fixed_z_, train_epoch):
    num_iter = 0
    collect_x_gen = []
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []

        epoch_start_time = time.time()
        for x_, _ in tqdm(train_loader):
            # train discriminator D
            D.zero_grad()

            mini_batch = x_.size()[0]

            #####
            # create y_real_ and y_fake_ tensors you will use in your BCE loss to push probabilities 
            # in the proper direction
            # y_real_ will be a tensor of all ones, because you want whatever is output by the generator
            # to be more likely to be real and you want the discriminator to recognize real images
            # y_fake_ then is a tensor of all zeros, because you want the discriminator to recognize fake images
            #####
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)

            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            

            #####
            # pass x_ through the decoder to get D_result
            # you will need to squeeze() the output before passing it to BCE_loss
            # compute D_real_loss using BCE_loss and the proper y tensor from above
            # you are trying to make the discriminator recognize the real image properly
            #####
            D_result = D(x_).squeeze()
            
            print('Shapes:')
            print('x_:',x_.shape)
            print('D_result:',D_result.shape)
            print('y_real_:',y_real_.shape)

            D_real_loss = BCE_loss(D_result, y_real_)

            #####
            # sample a z vector (remember to view(-1,100,1,1))
            # pass the z vector to the GPU and through your generator
            # this will create G_result
            #####

            z = torch.randn(mini_batch, 100).view(-1,100,1,1).cuda()
            G_result = G(z)

            #####
            # pass G_result through the discriminator and get D_result
            # you will need to squeeze() the output of the discriminator
            # compute D_fake_loss for the generated images by using BCE_loss and the proper y_tensor
            # you are trying to make the discriminator recognize the fake image properly
            # reduce D_fake_loss to the mean value
            #####

            D_result_fake = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result_fake, y_fake_)

            #####
            # sum D_real_loss and D_fake_loss to get D_train_loss
            # compute the gradients
            # step the optimizer
            #####

            D_train_loss = D_real_loss + D_fake_loss
            D_losses.append(D_train_loss.item())        

            D_train_loss.backward()
            D_optimizer.step()

            # train generator G
            G.zero_grad()

            #####
            # sample a z vector (viewed properly) and pass it to the GPU and through the generator
            # compute the discriminated value of the generated image, properly squeezing the output
            # get G_train_loss by using BCE_loss and the proper y_tensor
            # you are trying to make the generator generate real images
            # compute the gradients
            # step the optimizer
            #####

            z = torch.randn(mini_batch, 100).view(-1,100,1,1).cuda()
            G_result = G(z)
            D_result_G = D( G_result ).squeeze()

            G_train_loss = BCE_loss(D_result_G, y_real_)
            G_losses.append(G_train_loss.item())

            G_train_loss.backward()
            G_optimizer.step()

            num_iter += 1

            if num_iter > 500:
                break

        # generate a fixed_z_ image and save
        x_gen = G(fixed_z_)
        collect_x_gen.append(x_gen.detach().clone())
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('------------------------------------------------')
        # print out statistics
        print('Epoch [%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), 
                    train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                    torch.mean(torch.FloatTensor(G_losses))))
        show_imgs(G_result[:4])

    return [collect_x_gen, G, D, G_optimizer, D_optimizer, BCE_loss]

# %%
list_of_vars = train_gan(G, D, G_optimizer, D_optimizer, BCE_loss, train_loader, fixed_z_, train_epoch)
collect_x_gen, G, D, G_optimizer, D_optimizer, BCE_loss = list_of_vars

# %%
print( G(fixed_z_)[:4].shape)
show_imgs( G(fixed_z_)[:4] )

# %%
for x_gen in collect_x_gen:
    show_imgs(x_gen)

# %% [markdown]
# Now generate something with your own dataset! (Fashion, Mnist, Coco, Bedrooms, Pokemon)

# %%


# %% [markdown]
# # A demo of a state of the art GAN and "painting" with them in your browser:
# 
# https://gandissect.csail.mit.edu
# 
# By our colleagues at the MIT-IBM Watson AI Lab.
# 
# %% [markdown]
# # Where to go from here
# * Use a more exciting datasets - check out [the pytorch torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html) to get started quickly.
# * The [original GAN paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets)
# * The [DCGAN paper](https://arxiv.org/abs/1511.06434) which made it all work much better for images. Start from: pytorch DCGAN [example](https://github.com/pytorch/examples/blob/master/dcgan/main.py) and [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
# * Newer generations of loss functions measure different distances between distributions $p_{data}$ and $p_G$. For example [WGAN](https://github.com/martinarjovsky/WassersteinGAN), [WGAN-GP](https://arxiv.org/abs/1704.00028), [Fisher GAN](https://arxiv.org/abs/1705.09675), [Sobolev GAN](https://arxiv.org/abs/1711.04894), many more. They often have better stability properties wrt the original GAN loss.
# 
# # References for this tutorial
# * pytorch DCGAN [example](https://github.com/pytorch/examples/blob/master/dcgan/main.py) and [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) by Nathan Inkawhich
# * [Medium blog post](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f) by Diego Gomez Mosquera
# * [Material made for ITDS course at CUNY](https://github.com/grantmlong/itds2019/blob/master/lecture-6/DL_lab_solutions.ipynb) by Tom Sercu (that's me!)
# * [Blog post](https://towardsdatascience.com/graduating-in-gans-going-from-understanding-generative-adversarial-networks-to-running-your-own-39804c283399) by Cecelia Shao
# * [GAN overview image](https://www.slideshare.net/xavigiro/deep-learning-for-computer-vision-generative-models-and-adversarial-training-upc-2016) from Xavier Giro
