# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %%
import time
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch
import pdb
import re
import random
import string
import unidecode
from IPython.core.debugger import Pdb
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/sethmnielsen/cs474_labs_f2019/blob/master/DL_Lab6.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Lab 6: Sequence-to-sequence models
#
# ## Description:
# For this lab, you will code up the [char-rnn model of Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). This is a recurrent neural network that is trained probabilistically on sequences of characters, and that can then be used to sample new sequences that are like the original.
#
# This lab will help you develop several new skills, as well as understand some best practices needed for building large models. In addition, we'll be able to create networks that generate neat text!
#
# ## There are two parts of this lab:
# ###  1.   Wiring up a basic sequence-to-sequence computation graph
# ###  2.   Implementing your own GRU cell.
#
#
# An example of my final samples are shown below (more detail in the
# final section of this writeup), after 150 passes through the data.
# Please generate about 15 samples for each dataset.
#
# <code>
# And ifte thin forgision forward thene over up to a fear not your
# And freitions, which is great God. Behold these are the loss sub
# And ache with the Lord hath bloes, which was done to the holy Gr
# And appeicis arm vinimonahites strong in name, to doth piseling
# And miniquithers these words, he commanded order not; neither sa
# And min for many would happine even to the earth, to said unto m
# And mie first be traditions? Behold, you, because it was a sound
# And from tike ended the Lamanites had administered, and I say bi
# </code>
#
# %% [markdown]
# ---
#
# ## Part 0: Readings, data loading, and high level training
#
# ---
#
# There is a tutorial here that will help build out scaffolding code, and get an understanding of using sequences in pytorch.
#
# * Read the following
#
# > * [Pytorch sequence-to-sequence tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
# * [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
#
#
#
#
#

# %%
get_ipython().system(" wget -O ./text_files.tar.gz 'https://piazza.com/redirect/s3?bucket=uploads&prefix=attach%2Fjlifkda6h0x5bk%2Fhzosotq4zil49m%2Fjn13x09arfeb%2Ftext_files.tar.gz' ")
get_ipython().system(' tar -xzf text_files.tar.gz')
get_ipython().system(' pip install unidecode')
get_ipython().system(' pip install torch')


all_characters = string.printable
n_characters = len(all_characters)
file = unidecode.unidecode(open('./text_files/lotr.txt').read())
file_len = len(file)
print('file_len =', file_len)


# %%
chunk_len = 200


def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]


print(random_chunk())


# %%
# Turn string into list of longs


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)


print(char_tensor('abcDEF'))

# %% [markdown]
# ---
#
# ## Part 4: Creating your own GRU cell
#
# **(Come back to this later - its defined here so that the GRU will be defined before it is used)**
#
# ---
#
# The cell that you used in Part 1 was a pre-defined Pytorch layer. Now, write your own GRU class using the same parameters as the built-in Pytorch class does.
#
# Please try not to look at the GRU cell definition. The answer is right there in the code, and in theory, you could just cut-and-paste it. This bit is on your honor!
#
# **TODO:**
# * Create a custom GRU cell
#
# **DONE:**
#
#

# %%


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        full_input_size = input_size + hidden_size
        self.Wr = nn.Linear(full_input_size, hidden_size)
        self.Wz = nn.Linear(full_input_size, hidden_size)
        self.W = nn.Linear(full_input_size, hidden_size)

        self.sigmo = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Do stuff to make more layers here

    def forward(self, inputs, hidden):
        xt = inputs
        ht_prev = hidden
        rt = self.sigmo( self.Wr( torch.cat((xt, ht_prev),dim=2) ) )
        zt = self.sigmo( self.Wz( torch.cat((xt, ht_prev),dim=2) ) )
        rh_prev = torch.mul(rt, ht_prev)
        w = self.W ( torch.cat((rh_prev, xt),dim=2) )
        ht = torch.tanh(w)
        out_decoded = ht # check to make sure this is right
        hidden = ht
        return out_decoded, hidden




# %% [markdown]
# ---
#
# ##  Part 1: Building a sequence to sequence model
#
# ---
#
# Great! We have the data in a useable form. We can switch out which text file we are reading from, and trying to simulate.
#
# We now want to build out an RNN model, in this section, we will use all built in Pytorch pieces when building our RNN class.
#
#
# **TODO:**
# * Create an RNN class that extends from nn.Module.
#
# **DONE:**
#
#

# %%
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.gru = GRU(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_char, hidden):
        # by reviewing the documentation, construct a forward function that properly uses the output
        # of the GRU
        embedded = self.embedding(input_char).view(1,1,-1)
        output = embedded
        out_decoded, hidden = self.gru(output, hidden)
        return out_decoded, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


# %%
def random_training_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

# %% [markdown]
# ---
#
# ## Part 2: Sample text and Training information
#
# ---
#
# We now want to be able to train our network, and sample text after training.
#
# This function outlines how training a sequence style network goes.
#
# **TODO:**
# * Fill in the pieces.
#
# **DONE:**
#
#
#

# %%


def train(inp, target, decoder, decoder_optimizer, criterion):
    # initialize hidden layers, set up gradient and loss
    decoder_optimizer.zero_grad()
    hidden = decoder.init_hidden()
    loss = 0

    for char_in, char_target in zip(inp, target):
        char_decoded = decoder(char_in, hidden)
        loss += criterion(char_decoded.squeeze(0), char_target.unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    loss_score = loss.item()/len(inp)
    return loss_score

# %% [markdown]
# ---
#
# ## Part 3: Sample text and Training information
#
# ---
#
# You can at this time, if you choose, also write out your train loop boilerplate that samples random sequences and trains your RNN. This will be helpful to have working before writing your own GRU class.
#
# If you are finished training, or during training, and you want to sample from the network you may consider using the following function. If your RNN model is instantiated as `decoder`then this will probabilistically sample a sequence of length `predict_len`
#
# **TODO:**
# * Fill out the evaluate function to generate text frome a primed string
#
# **DONE:**
#
#

# %%


def evaluate(decoder=None, prime_str='A', predict_len=100, temperature=0.8):
    # initialize hidden variable, initialize other useful variables
    decoder.init_hidden()
    prime_chars = char_tensor(prime_str)

    for k in range( len(prime_chars)-1 ):
        out, hidden = decoder(prime_chars[i], hidden)
    eval_char = prime_chars[-1]

    prediction = prime_str
    for k in range(predict_len):
        out, hidden = decoder(eval_char, )
        prob_distrib = out.data.view(-1).div(temperature).exp()
        select_ind = torch.multinomial(prob_distrib, 1)[0]

        char_choice = all_characters[select_ind]
        eval_char = char_tensor(char_choice)
        prediction += char_choice

    return prediction

# %% [markdown]
# ---
#
# ## Part 4: (Create a GRU cell, requirements above)
#
# ---
#
# %% [markdown]
#
# ---
#
# ## Part 5: Run it and generate some text!
#
# ---
#
# Assuming everything has gone well, you should be able to run the main function in the scaffold code, using either your custom GRU cell or the built in layer, and see output something like this. I trained on the “lotr.txt” dataset, using chunk_length=200, hidden_size=100 for 2000 epochs gave.
#
# **TODO:**
# * Create some cool output
#
# **DONE:**
#
#

# %%
n_epochs = 5000
print_every = 200
plot_every = 10
hidden_size = 200
n_layers = 1
lr = 0.001

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

# %%
for epoch in range(1, n_epochs + 1):
    loss_ = train( *random_training_set(), decoder, decoder_optimizer, criterion)
    loss_avg += loss_

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' %
              (time.time() - start, epoch, epoch / n_epochs * 100, loss_))
        print(evaluate('Wh', 100), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0


# %%
for i in range(10):
    start_strings = [" Th", " wh", " he", " I ", " ca", " G", " lo", " ra"]
    start = random.randint(0, len(start_strings)-1)
    print(start_strings[start])
#   all_characters.index(string[c])
    print(evaluate(start_strings[start], 200), '\n')

# %% [markdown]
# ---
#
# ## Part 6: Generate output on a different dataset
#
# ---
#
# **TODO:**
#
# * Choose a textual dataset. Here are some [text datasets](https://www.kaggle.com/datasets?tags=14104-text+data%2C13205-text+mining) from Kaggle
#
# * Generate some decent looking results and evaluate your model's performance (say what it did well / not so well)
#
# **DONE:**
#
#
