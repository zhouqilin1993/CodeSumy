# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import setting
from buildVocab import readVocab,variablesPairsFromData
from evaluate import evaluateRandomly
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from train import trainIters

import torch

# Training
def trainDemo(lang, dataSet, nlVocab, codeVocab, train_variables):
    print("Training...")
    encoder1 = EncoderRNN(codeVocab.n_words, setting.HIDDDEN_SIAZE)
    attn_decoder1 = AttnDecoderRNN(setting.HIDDDEN_SIAZE, nlVocab.n_words, 1, dropout_p=0.1)

    if setting.USE_CUDA:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    trainIters(lang, dataSet, train_variables, encoder1, attn_decoder1, 2000000, print_every=5000)

# Evaluating
def evalDemo(lang, dataSet, nlVocab, codeVocab, test_pairs):
    print("Evaluating...")
    encoder = torch.load(setting.MODEL_HOME + "/%s.%s.encoder.pkl" % (dataSet, lang))
    decoder = torch.load(setting.MODEL_HOME + "/%s.%s.decoder.pkl" % (dataSet, lang))

    evaluateRandomly(lang, dataSet, encoder, decoder, test_pairs)

    # Test a demo and visualizing Attention
    # print("Visualizing Attention...")

    # codeInput = "je suis trop froid ."
    # output_words, attentions = evaluate(nlVocab, codeVocab, encoder, decoder, codeInput)
    # print("type",type(attentions))
    # plt.matshow(attentions.numpy())

    # evaluateAndShowAttention(nlVocab, codeVocab, encoder, decoder,codeInput)

def trainAndEval(lang, dataSet, ifTrain):
    nlVocab, codeVocab = readVocab(lang, dataSet)
    train_pairs, train_variables = variablesPairsFromData("train", lang, dataSet)
    test_pairs, test_variables = variablesPairsFromData("test", lang, dataSet)
    #valid_pairs, valid_variables = variablesPairsFromData("valid", lang, dataSet)

    if ifTrain:
        trainDemo(lang, dataSet, nlVocab, codeVocab, train_variables)
    evalDemo(lang, dataSet, nlVocab, codeVocab, test_pairs)


if __name__ == '__main__':
    trainAndEval('java', 'so', True)
    trainAndEval('csharp', 'so', True)





