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
from IPython import get_ipython
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
# %%
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()

    def forward(self, inputs, hidden):
        # Each layer does the following:
        # r_t = sigmoid(W_ir*x_t + b_ir + W_hr*h_(t-1) + b_hr)
        # z_t = sigmoid(W_iz*x_t + b_iz + W_hz*h_(t-1) + b_hz)
        # n_t = tanh(W_in*x_t + b_in + r_t**(W_hn*h_(t-1) + b_hn))
        # h_(t) = (1 - z_t)**n_t + z_t**h_(t-1)
        # Where ** is hadamard product (not matrix multiplication, but elementwise multiplication)

        return outputs, hiddens

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

# %%
def evaluate(prime_str='A', predict_len=100, temperature=0.8, decoder=None):
    # initialize hidden variable, initialize other useful variables
    decoder.init_hidden()


n_epochs = 5000
print_every = 200
plot_every = 10
hidden_size = 200
n_layers = 3
lr = 0.001

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0


# %%
# n_epochs = 2000
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
