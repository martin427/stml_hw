from io import open
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

import torch
print(torch.__version__)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        #print("input_size: ", input_size)
        #print("hidden_size: ", hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        #self.embedding.weight.data.copy_(torch.eye(hidden_size))
        #self.embedding.weight.requires_grad = False
        
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

class DecoderRNN(nn.Module):###直接在這裏面output gru的reset gate和 update gate
    
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.embedding.weight.data.copy_(torch.eye(hidden_size))
        #self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        #output = self.embedding(input).view(1, batch_size, self.hidden_size)
        #for i in range(self.n_layers):
        #    output = F.relu(output)
        #    output, hidden = self.gru(output, hidden)
        input_embedding = self.embedding(input).view(1, batch_size, self.hidden_size)
        relu_input = F.relu(input_embedding)
        output, hidden_out = self.gru(relu_input, hidden)

        print("self.gru.weight_ih_l0: ", self.gru.weight_ih_l0)
        print("relu_input: ", relu_input)
        print("input_embedding.shape: ", input_embedding.shape)#torch.Size([1, 128, 150])
        print("relu_input.shape: ", relu_input.shape)#torch.Size([1, 128, 150])
        print('gru.weight_ih_l0.size()', self.gru.weight_ih_l0.size())  # torch.Size([450, 150])
        #(W_ir|W_iz|W_in), of shape (3*hidden_size x input_size)
        print('gru.weight_hh_l0.size()', self.gru.weight_hh_l0.size())
        #(W_hr|W_hz|W_hn), of shape (3*hidden_size x hidden_size)
        print('gru.bias_ih_l0.size()', self.gru.bias_ih_l0.size())#torch.Size([450])
        # (b_ir|b_iz|b_in), of shape (3*hidden_size)
        print('gru.bias_hh_l0.size()', self.gru.bias_hh_l0.size())#torch.Size([450])
        #b_hr|b_hz|b_hn), of shape (3*hidden_size)
        w_ih = self.gru.weight_ih_l0
        b_ih = self.gru.bias_ih_l0
        w_hh =  self.gru.weight_hh_l0
        b_hh = self.gru.bias_hh_l0
        print("type(w_ih): ", type(w_ih))
        print("type(b_ih): ", type(b_ih))
        print("type(w_hh): ", type(w_hh))
        print("type(b_hh): ", type(b_hh))
        print("type(input): ", type(input))
        relu_input = torch.squeeze(relu_input)
        hidden = torch.squeeze(hidden)
        #gi = F.linear(input, w_ih, b_ih)
        gi = F.linear(relu_input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        print("gi.size(): ", gi.size())
        print("gh.size(): ", gh.size())
        print("i_r.size(): ", i_r.size())
        print("h_r.size(): ", h_r.size())
        print("i_i.size(): ", i_i.size())
        print("h_i.size(): ", h_i.size())
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        print("resetgate")
        print(resetgate)
        print("inputgate")
        print(inputgate)
        print("newgate")
        print(newgate)
        print("hy")
        print(hy)
        print("resetgate.size(): ", resetgate.size())
        print("inputgate.size(): ", inputgate.size())
        print("newgate.size(): ", newgate.size())
        print("hy.size(): ", hy.size())

        exit()

        output = self.softmax(self.out(output[0]))
        return output, hidden_out, self.gru

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    batch_size = input_variable.size()[0]
    encoder_hidden = encoder.initHidden(batch_size)
    input_variable = Variable(input_variable.transpose(0, 1))
    target_variable = Variable(target_variable.transpose(0, 1))
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], batch_size, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    #if use_teacher_forcing:#
    if False:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, batch_size, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
            #print("di: ", di)
            #print("decoder_input: ", decoder_input)
            #print()
            
    else:
        # Without teacher forcing: use its own predictions as the next input
        #print("for di in range(target_length):")
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder( #return output, hidden, self.gru
                decoder_input, batch_size, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, target_variable[di])
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def trainIters(encoder, decoder, epochs, train_loader, test_loader, max_length, learning_rate=0.01):
    start = time.time()
    number_of_batch = 0
    encoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, encoder.parameters()),
                                  lr=learning_rate)
    decoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, decoder.parameters()),
                                  lr=learning_rate)

    # data loader
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        print_loss_total = 0
        accu = 0
        for batch_x, batch_y in train_loader:
            #print("batch_x")
            #print(batch_x)
            #print("batch_y")
            #print(batch_y)
            #print("batch_x.shape: ", batch_x.shape)
            #print("batch_y.shape: ", batch_y.shape)
            #print("batch_x.shape[0]: ", batch_x.shape[0])
            #print("batch_y.shape[0]: ", batch_y.shape[0])

            accu = accu + batch_x.shape[0]
            #print("accu: ", accu)
            loss = train(batch_x, batch_y, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)            
            print_loss_total += loss
            print('training batch loss: ' + str(loss))
            number_of_batch = number_of_batch + 1
            if(number_of_batch % 5 == 0):
                print('epochs: ' + str(epoch))
                print('in validation training batch loss: ' + str(loss))
                evaluation_loss = evaluate(encoder, decoder, test_loader, max_length)
                print('evaluation_loss: ' + str(evaluation_loss))
                exit()

        print('epochs: '+str(epoch))
        print('total loss: '+str(print_loss_total))
        evaluate(encoder, decoder, test_loader, max_length)
        print()
        if(epoch%4 == 0):
            encoder1_model_path = "./save_model/rhyme/" + str(epoch) + "_1201_encoder1_rhyme.pt"
            attn_decoder1_model_path = "./save_model/rhyme/" + str(epoch) +"_1201_decoder1_rhyme.pt"
            with open(encoder1_model_path, 'wb') as f:
                torch.save(encoder1, f)
            with open(attn_decoder1_model_path, 'wb') as f:
                torch.save(decoder1, f)

def evaluate(encoder, decoder, loader, max_length):

    total = 0
    correct = 0
    criterion = nn.NLLLoss()
    loss = 0
    target_length = 5.0
    for batch_x, batch_y in loader:
        batch_size = batch_x.size()[0]
        encoder_hidden = encoder.initHidden(batch_size)
        input_variable = Variable(batch_x.transpose(0, 1))
        target_variable = Variable(batch_y.transpose(0, 1))
        #print("input_variable.shape: ", input_variable.shape)
        #print("target_variable.shape: ", target_variable.shape)
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], batch_size, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        for di in range(target_length):
            decoder_output, decoder_hidden, gru= decoder(
                decoder_input, batch_size, decoder_hidden)
            #print("gru.weight_ih_l0")###只有weight_ih_l0可以拿來用，11, 12全部都不能用
            #print(gru.weight_ih_l0)
            print("di: ", di)
            print("decoder_input")
            print(decoder_input)
            print()

            print('decoder_input.size()', decoder_input.size())#torch.Size([128])
            print("type(decoder_input): ", type(decoder_input))#
            print("decoder_output")
            print(decoder_output)
            print('decoder_output.size()', decoder_output.size())#torch.Size([128, 91710])
            print("type(gru.weight_ih_l0): ", type(gru.weight_ih_l0))#<class 'torch.nn.parameter.Parameter'>
            print('gru.weight_ih_l0.size()', gru.weight_ih_l0.size())#torch.Size([384, 128])
            print('gru.weight_hh_l0.size()', gru.weight_hh_l0.size())
            print('gru.bias_ih_l0.size()', gru.bias_ih_l0.size())
            print('gru.bias_hh_l0.size()', gru.bias_hh_l0.size())
            #print("gru.state.state_dict(): ", gru.state_dict())
            print("gru.state.state_dict().keys(): ", gru.state_dict().keys())
            #['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
            exit()

            topv, topi = decoder_output.data.topk(1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, target_variable[di])
        exit()
        """
        print("validation_bacth_count: ", validation_bacth_count)
        batch_size = batch_x.size()[0]
        print("batch_size: ", batch_size)
        encoder_hidden = encoder.initHidden(batch_size)

        input_variable = Variable(batch_x.transpose(0, 1))
        target_variable = Variable(batch_y.transpose(0, 1))

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        print("input_variable.shape: ", input_variable.shape)
        print("target_variable.shape: ", target_variable.shape)
        #output = torch.LongTensor(target_length, batch_size)

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
            decoder_output, decoder_hidden= decoder(
                decoder_input, batch_size, decoder_hidden)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, target_variable[di])
        validation_bacth_count = validation_bacth_count + 1
        """
    return loss.item() / target_length
    #print('accuracy '+str(correct/total))

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 20#20
teacher_forcing_ratio = 0.5
hidden_size = 150#a-z+SOS+EOS+PAD  以前是128
batch_size = 128#128
epochs = 100#15
###testing

#
input_lang, output_lang, pairs = prepareData('eng', 'fra', False)#prepareData('eng', 'fra', True)
pairs = variablesFromPairs(input_lang, output_lang, pairs, MAX_LENGTH)
train_num = int(len(pairs)*0.9)
#print("train_num: ", train_num)#122257
#print("len(pairs)-train_num: ", len(pairs)-train_num)#13585
#print("pairs")
#print(pairs)
print("pairs[0]: ", pairs[0])
print("pairs[0][0]: ", pairs[0][0])
print("pairs[0][1]: ", pairs[0][1])
print("pairs[0][0].shape: ", pairs[0][0].shape)
print("pairs[0][0].shape[0]: ", pairs[0][0].shape[0])
print("type(pairs): ", type(pairs))
print("len(pairs): ", len(pairs))
count = 0
for i in range(len(pairs)):
    for j in range(len(pairs[0])):
        length = pairs[i][j].shape[0]
        if(length != MAX_LENGTH):
            count = count + 1
            print("j: ", j)
            print("pairs[0][j]: ", pairs[i][j])
            print("length: ", length)
            print()
print("len(pairs): ", len(pairs))
print("count: ", count)

#print("input_lang.word2index")
#print(input_lang.word2index)

train_loader = torch.utils.data.DataLoader(pairs[:train_num], 
    batch_size=batch_size, shuffle=True)#shuffle=True
test_loader = torch.utils.data.DataLoader(pairs[train_num:], 
    batch_size=batch_size, shuffle=False)#shuffle=True

#pickle.dump(output_lang, open("./save_model/output_lang.p", "wb"))
#output_lang = pickle.load(open("./save_model/output_lang.p", "rb"))
pickle.dump(output_lang, open("../save_model/output_lang_length.p", "wb"))
output_lang = pickle.load(open("../save_model/output_lang_length.p", "rb"))

#encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
encoder1 = EncoderRNN(output_lang.n_words, hidden_size)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words)
if use_cuda:
    encoder1 = encoder1.cuda()
    #decoder1 = attn_decoder1.cuda()
    decoder1 = decoder1.cuda()
print('Training starts.')
trainIters(encoder1, decoder1, epochs, train_loader, test_loader, MAX_LENGTH)

encoder1_model_path = "../save_model/1201_encoder1_length.pt"
attn_decoder1_model_path = "../save_model/1201_decoder1_length.pt"
with open(encoder1_model_path, 'wb') as f:
    torch.save(encoder1, f)
with open(attn_decoder1_model_path, 'wb') as f:
    torch.save(decoder1, f)

