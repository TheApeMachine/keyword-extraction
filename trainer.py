# Based on the PyTorch tutorial for translation.
# Original code by: Sean Robertson
#                   https://github.com/spro/practical-pytorch

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import sys

use_cuda = False
TRAIN    = False

for arg in sys.argv:
    if arg == '--train':
        TRAIN = True
    elif arg == '--cuda':
        use_cuda = torch.cuda.is_available()

print("CUDA : ", use_cuda)
print("TRAIN: ", TRAIN)

SOS_token = 0
EOS_token = 1

class Lang:

    def __init__(self, name):
        self.name       = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words    = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word]          = self.n_words
            self.word2count[word]          = 1
            self.index2word[self.n_words]  = word
            self.n_words                  += 1
        else:
            self.word2count[word] += 1

def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-z_AZ.!?]+", r" ", s)

    return s

def read_langs(lang1, lang2, reverse=False):
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs       = [list(reverse(p)) for p in pairs]
        input_lang  = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang  = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 512

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    pairs                          = filter_pairs(pairs)

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('keyword', 'data', False)

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(input_size, hidden_size)
        self.gru         = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded       = self.embedding(input).view(1, 1, -1)
        output         = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(output_size, hidden_size)
        self.gru         = nn.GRU(hidden_size, hidden_size)
        self.out         = nn.Linear(hidden_size, output_size)
        self.softmax     = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output         = self.embedding(input).view(1, 1, -1)
        output         = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output         = self.softmax(self.out(output[0]))

        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.dropout_p    = dropout_p
        self.max_length   = max_length
        self.embedding    = nn.Embedding(self.output_size, self.hidden_size)
        self.attn         = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout      = nn.Dropout(self.dropout_p)
        self.gru          = nn.GRU(self.hidden_size, self.hidden_size)
        self.out          = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded       = self.embedding(input).view(1, 1, -1)
        embedded       = self.dropout(embedded)
        attn_weights   = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied   = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output         = torch.cat((embedded[0], attn_applied[0]), 1)
        output         = self.attn_combine(output).unsqueeze(0)
        output         = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output         = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        if use_cuda:
            return result.cuda()
        else:
            return result

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)

    result = Variable(torch.LongTensor(indexes).view(-1, 1))

    if use_cuda:
        return result.cuda()
    else:
        return result

def variables_from_pair(pair):
    input_variable  = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])

    return (input_variable, target_variable)

teacher_forcing_ratio = 0.5

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length    = input_variable.size()[0]
    target_length   = target_variable.size()[0]
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    loss            = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei]            = encoder_output[0][0]

    decoder_input  = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input  = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            loss          += criterion(decoder_output, target_variable[di])
            decoder_input  = target_variable[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            topv, topi     = decoder_output.data.topk(1)
            ni             = topi[0][0]
            decoder_input  = Variable(torch.LongTensor([[ni]]))
            decoder_input  = decoder_input.cuda() if use_cuda else decoder_input
            loss          += criterion(decoder_output, target_variable[di])

            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

import time
import math

def as_minutes(s):
    m  = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s   = now - since
    es  = s / (percent)
    rs  = es - s

    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def train_interations(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    start            = time.time()
    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs    = [variables_from_pair(random.choice(pairs)) for i in range(n_iters)]
    criterion         = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair   = training_pairs[iter - 1]
        input_variable  = training_pair[0]
        target_variable = training_pair[1]

        loss = train(
            input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
        )

        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg   = print_loss_total / print_every
            print_loss_total = 0

            print('%s (%d %d%%) %.4f' % (
                time_since(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg
            ))

import numpy as np

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable  = variable_from_sentence(input_lang, sentence)
    input_length    = input_variable.size()[0]
    encoder_hidden  = encoder.init_hidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_ouputs.cuda() if use_cuda else encoder_ouputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei]            = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input      = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input      = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden     = encoder_hidden
    decoded_words      = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        decoder_attentions[di] = decoder_attention.data
        topv, topi             = decoder_output.data.topk(1)
        ni                     = topi[0][0]

        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

hidden_size = 256

if TRAIN is True:
    print("TRAINING...")

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    train_interations(encoder1, attn_decoder1, 75000, print_every=5000)

    torch.save(encoder1, 'encoder.pt')
    torch.save(attn_decoder1, 'decoder.pt')
else:
    print("LOADING...")

    encoder1      = torch.load('encoder.pt')
    attn_decoder1 = torch.load('decoder.pt')

def output_evaluation(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence
    )

    print("input  = ", input_sentence)
    print("output = ", ' '.join(output_words))

while(True):
    try:
        inp = raw_input(">")
        output_evaluation(inp)
    except KeyError:
        pass
