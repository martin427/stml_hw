from io import open
import argparse
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch import optim
import torch.nn.functional as F
from helper import *
import pickle
import numpy as np
import sys

import torch
print(torch.__version__)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # print("input_size: ", input_size)
        # print("hidden_size: ", hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        # self.embedding.weight.data.copy_(torch.eye(hidden_size))
        # self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        # self.embedding.weight.data.copy_(torch.eye(hidden_size))
        # self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        output = self.embedding(input).view(1, batch_size, self.hidden_size)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        #print("output")
        #print(output)
        #for i in range(10):
        #    print("output[i][3]: ", output[i][3])
        #    print("output[i][4]: ", output[i][4])
        #    print("output[i][5]: ", output[i][5])
        #    print()
        #print("output.shape: ", output.shape)#torch.Size([128, 91712])  torch.Size([batch_size, vocab_size])
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def evaluate(encoder, decoder, loader, max_length, output_lang, output_path):
    total = 0
    correct = 0
    all_input = []
    all_output = []
    print("evaluate")
    for batch_x in loader:

        #print("batch_x")
        #print(batch_x)
        #print("batch_y")
        #print(batch_y)
        #print("batch_x.shape: ", batch_x.shape)#batch_x.shape:  torch.Size([128, 20])
        #print("batch_y.shape: ", batch_y.shape)#batch_y.shape:  torch.Size([128, 20])
        np_batch_x = batch_x.data.cpu().numpy().tolist()
        all_input = all_input + np_batch_x

        batch_size = batch_x.size()[0]
        encoder_hidden = encoder.initHidden(batch_size)

        input_variable = Variable(batch_x.transpose(0, 1))
        #target_variable = Variable(batch_y.transpose(0, 1))

        input_length = input_variable.size()[0]
        #target_length = target_variable.size()[0]
        target_length = max_length
        output = torch.LongTensor(target_length, batch_size)

        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], batch_size, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, batch_size, decoder_hidden)
            topv, topi = decoder_output.data.topk(1, dim=1)
            #print("topi")
            #print(topi)
            #print("topi.shape: ", topi.shape)

            # output[di] = torch.cat(topi)
            # decoder_input = torch.cat(topi)
            #decoder_input = Variable(topi)
            output[di] = topi.squeeze().detach()
            decoder_input = topi.squeeze().detach()  # detach from history as input
            #print("decoder_input")
            #print(decoder_input)
            #print()

        #print("output")
        #print(output)
        """output
        tensor([[ 3,  3,  3,  ...,  3,  3,  3],
        [ 9, 11, 11,  ...,  9,  9, 11],
        [ 9,  9, 11,  ...,  9,  9, 11],
        ...,
        [ 2,  2,  1,  ...,  2,  2,  9],
        [ 2,  2,  2,  ...,  2,  2,  1],
        [ 2,  2,  2,  ...,  2,  2,  2]])
        """
        #print("before transpose output.shape: ", output.shape)  # torch.Size([20, 128])

        output = output.transpose(0, 1)
        #print("after output")
        #print(output)
        #print("after transpose output.shape: ", output.shape)  # torch.Size([128, 20])
        np_output = output.data.numpy().tolist()
        all_output = all_output + np_output

        """
        for di in range(output.size()[0]):
            ignore = [SOS_token, EOS_token, PAD_token]
            sent = [word for word in output[di] if word not in ignore]
            y = [word for word in batch_y[di] if word not in ignore]
            if sent == y:
                correct += 1
            total += 1
        """
    all_input = np.asarray(all_input)
    all_output = np.asarray(all_output)
    print("all_input")
    print(all_input)
    print("all_output")
    print(all_output)
    print("all_input.shape: ", all_input.shape)
    print("all_output.shape: ", all_output.shape)
    all_word_output = []
    for i in range(all_output.shape[0]):
        temp_word_output = []
        string = ""
        for j in range(all_output.shape[1]):
            index = all_output[i][j]
            word = output_lang.index2word[index]
            if(word == 'SOS'):
                continue
            elif(word == 'EOS'):
                break
            #temp_word_output.append(word)
            string = string + word + " "
        string = string[:-1]
        all_word_output.append(string)

    #output_path = "./noatten_output_pos.csv"
    with open(output_path, 'w') as f:
        for i in range(len(all_word_output)):
            string = all_word_output[i]
            f.write('%s\n' % (string))
    exit()
    # print('accuracy '+str(correct/total))


input_file_path = sys.argv[1]
print("input_file_path: ", input_file_path)
output_file_path = sys.argv[2]
print("output_file_path: ", output_file_path)
vocab_file_path = sys.argv[3]
print("vocab_file_path: ", vocab_file_path)
encoder1_model_path = sys.argv[4]
print("encoder1_model_path: ", encoder1_model_path)
attn_decoder1_model_path = sys.argv[5]
print("attn_decoder1_model_path: ", attn_decoder1_model_path)

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 20#20
teacher_forcing_ratio = 0.5
hidden_size = 128 #a-z+SOS+EOS+PAD
batch_size = 128#128
epochs = 100#15
###testing

#
print("new")

input_test = test_readLangs('test', input_file_path)
#print("input_test")
#print(input_test)
#output_lang = pickle.load(open("./save_model/length/output_lang.p", "rb"))
#output_lang = pickle.load(open("./save_model/pos/output_lang_pos.p", "rb"))
print("vocab_file_path: ", vocab_file_path)
#output_lang = pickle.load(open(vocab_file_path, "rb"))
output_lang = pickle.load(open(vocab_file_path, "rb"))
print("before variablesFromtest")
test_input = variablesFromtest(output_lang, input_test, MAX_LENGTH)
print("after variablesFromtest")
#print("test_input")
#print(test_input)
print("len(input_test): ", len(input_test))#len(input_test):  70000
print("len(input_test[0]): ", len(input_test[0]))#22
print("len(test_input): ", len(test_input))#70000
print("len(test_input[0]): ", len(test_input[0]))#20
print("before test_loader")
test_loader = torch.utils.data.DataLoader(test_input,
    batch_size=batch_size, shuffle=False)
print("after test_loader")
#encoder1_model_path = "./save_model/44_1201_encoder1_length.pt"
#attn_decoder1_model_path = "./save_model/44_1201_decoder1_length.pt"
#encoder1_model_path = "./save_model/pos/44_1201_encoder1_pos.pt"
#attn_decoder1_model_path = "./save_model/pos/44_1201_decoder1_pos.pt"
encoder1_model_path = encoder1_model_path
attn_decoder1_model_path = attn_decoder1_model_path
encoder1 = torch.load(encoder1_model_path)
attn_decoder1 = torch.load(attn_decoder1_model_path)
evaluate(encoder1, attn_decoder1, test_loader, MAX_LENGTH, output_lang, output_file_path)
exit()
input_lang, output_lang, pairs = prepareData('eng', 'fra', False)#prepareData('eng', 'fra', True)
pairs = variablesFromPairs(input_lang, output_lang, pairs, MAX_LENGTH)
train_num = int(len(pairs)*0.9)

train_loader = torch.utils.data.DataLoader(pairs[:train_num],
    batch_size=batch_size, shuffle=False)#shuffle=True
test_loader = torch.utils.data.DataLoader(pairs[train_num:],
    batch_size=batch_size, shuffle=False)#shuffle=True

output_lang = pickle.load(open("./save_model/output_lang.p", "rb"))
"""
print("output_lang.index2word[0]: ", output_lang.index2word[0])
print("output_lang.index2word[1]: ", output_lang.index2word[1])
print("output_lang.index2word[2]: ", output_lang.index2word[2])
print("output_lang.index2word[3]: ", output_lang.index2word[3])
print("output_lang.word2index[SOS]: ", output_lang.word2index["SOS"])
print("output_lang.word2index[EOS]: ", output_lang.word2index["EOS"])
"""
#print("output_lang.word2index[PAD]: ", output_lang.word2index["PAD"])

encoder1_model_path = "./save_model/12_1201_encoder1_length.pt"
attn_decoder1_model_path = "./save_model/12_1201_decoder1_length.pt"
encoder1 = torch.load(encoder1_model_path)
attn_decoder1 = torch.load(attn_decoder1_model_path)
evaluate(encoder1, attn_decoder1, test_loader, MAX_LENGTH)
