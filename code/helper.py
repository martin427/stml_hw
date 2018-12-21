import unicodedata
import re
import torch
from torch.autograd import Variable
import time
import math
import random
#training_path = "./data/hw3_1/length/train.csv"
training_path = "../data/original_data/train.csv"
#testing_path = "./data/hw3_1/length/test.csv"
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 20#20
use_cuda = torch.cuda.is_available()

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "PAD": 2}
        self.word2count = {"SOS": 1, "EOS": 1, "PAD": 1}
        self.index2word = {0: "SOS", 1: "EOS", 2: 'PAD'}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        #print("sentence: ", sentence)
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        #print("self.word2index: ", self.word2index)
        #print("word: ", word)
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    #lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').        read().strip().split('\n')
    lines = open(training_path, encoding='utf-8'). \
        read().strip().split('\n')
    # Split every line into pairs and normalize
    #pairs = [[s for s in l.split('\t')] for l in lines]
    pairs = [[s for s in l.split(',')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def test_readLangs(lang1, testing_path, reverse=False):
    lines = open(testing_path, encoding='utf-8'). \
        read().strip().split('\n')
    return lines

def filterPair(p):
    #return len(p[0].split(' ')) < MAX_LENGTH and \
    #    len(p[1].split(' ')) < MAX_LENGTH and \
    #    p[1].startswith(eng_prefixes)
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, max_length, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        #input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    index_list = []
    components_sentence = sentence.split(' ')
    #print("lang.index2word[80]: ", lang.index2word[80])#昨天

    for word in components_sentence:
        if(word in lang.word2index):
            #print("lang.word2index[word]: ", lang.word2index[word])
            #print("type(lang.word2index[word]): ", type(lang.word2index[word]))
            index_list.append(lang.word2index[word])
        else:
            index_list.append(80)
    return index_list
    #return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence, max_length):
    #print("sentence")#SOS 灵魂 不再 悲戚 EOS 6 NOR
    #print(sentence)
    indexes = indexesFromSentence(lang, sentence)

    #print("indexes")#[0, 459, 1122, 23902, 1, 21, 12]
    #print(indexes)

    #print("before append extend indexes")
    #print(indexes)

    indexes.append(EOS_token)
    indexes.extend([PAD_token] * (max_length - len(indexes)))
    #print("after append extend indexes")
    #print(indexes)


    result = torch.LongTensor(indexes)
    if use_cuda:
        return result.cuda()
    else:
        return result

def TestVariableFromSentence(lang, sentence, max_length):
    #print("sentence")#SOS 灵魂 不再 悲戚 EOS 6 NOR
    #print(sentence)
    indexes = indexesFromSentence(lang, sentence)
    if(len(indexes) > max_length-1):
        #print("indexes: ", indexes)
        #print("len(indexes): ", len(indexes))
        #print("indexes[-3:]: ", indexes[-3:])
        last_three_element = indexes[-3:]#[1, 1562, 12]
        constrained_indexes = indexes[:19]
        #print("constrained_indexes: ", constrained_indexes)
        #print("len(constrained_indexes): ", len(constrained_indexes))
        constrained_indexes[-3:] = last_three_element
        #print("new constrained_indexes: ", constrained_indexes)
        #print("new len(constrained_indexes): ", len(constrained_indexes))

        indexes = constrained_indexes
    #print("indexes")#[0, 459, 1122, 23902, 1, 21, 12]
    #print(indexes)

    #print("before append extend indexes")
    #print(indexes)

    indexes.append(EOS_token)
    indexes.extend([PAD_token] * (max_length - len(indexes)))
    #print("after append extend indexes")
    #print(indexes)


    result = torch.LongTensor(indexes)
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPairs(input_lang, output_lang, pairs, max_length):
    res = []
    for pair in pairs:
        #input_variable = variableFromSentence(input_lang, pair[0], max_length)
        input_variable = variableFromSentence(output_lang, pair[0], max_length)
        target_variable = variableFromSentence(output_lang, pair[1], max_length)
        res.append((input_variable, target_variable))
    return res

def variablesFromtest(output_lang, input_test, max_length):
    res = []
    for temp_test in input_test:
        input_variable = TestVariableFromSentence(output_lang, temp_test, max_length)
        res.append(input_variable)
    return res

