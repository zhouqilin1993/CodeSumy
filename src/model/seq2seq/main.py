# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import random
import setting
import matplotlib.pyplot as plt
from buildVocab import readVocab,variablesPairsFromData
from evaluate import evaluateRandomly,evaluate
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from train import trainIters
from utils import evaluateAndShowAttention

import torch

lang = 'java'
dataSet = 'so'
dataType = 'train'

# Training
print("Training...")
nlVocab, codeVocab = readVocab(lang, dataSet)
train_pairs, train_variables = variablesPairsFromData("train", lang, dataSet)
test_pairs, test_variables = variablesPairsFromData("test", lang, dataSet)
valid_pairs, valid_variables = variablesPairsFromData("valid", lang, dataSet)

encoder1 = EncoderRNN(codeVocab.n_words, setting.HIDDDEN_SIAZE)
attn_decoder1 = AttnDecoderRNN(setting.HIDDDEN_SIAZE, nlVocab.n_words, 1, dropout_p=0.1)

# torch.save(encoder1, setting.MODEL_HOME + "/%s.%s.encoder.pkl" % (dataSet, lang))
# torch.save(attn_decoder1,setting.MODEL_HOME + "/%s.%s.decoder.pkl" % (dataSet, lang))
#
# encoder = torch.load(setting.MODEL_HOME + "/%s.%s.encoder.pkl" % (dataSet, lang))
# decoder = torch.load(setting.MODEL_HOME + "/%s.%s.decoder.pkl" % (dataSet, lang))

if setting.USE_CUDA:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(lang, dataSet, train_variables, encoder1, attn_decoder1, 75000, print_every=5000)

# Evaluating
print("Evaluating...")
evaluateRandomly(lang, dataSet, encoder1, attn_decoder1,test_pairs)
# Test a demo and visualizing Attention
print("Visualizing Attention...")

codeInput = "je suis trop froid ."
output_words, attentions = evaluate(nlVocab, codeVocab, encoder1, attn_decoder1, codeInput)
plt.matshow(attentions.numpy())

evaluateAndShowAttention(nlVocab, codeVocab, encoder1,attn_decoder1,codeInput)


