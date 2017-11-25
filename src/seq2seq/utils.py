# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
import setting
from evaluate import evaluate
from buildVocab import readVocab

import torch
from ..metrics.metric import metricPair

def getMetirc(ref_str, gen_str):
    return metricPair(ref_str, gen_str)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(setting.PNG_HOME + time.strftime("/[%Y%m%d %H:%M:%S].png", time.localtime()))

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+0.01)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showAttention(attentions, input_sentence, output_words ):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print("atten:", attentions)
    cax = ax.matshow(torch.FloatTensor(attentions).numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig(setting.PNG_HOME + time.strftime("/[%Y%m%d %H:%M:%S]attention.png", time.localtime()))


def evalDemo(dataSet, lang, CodeStr):
    nlVocab, codeVocab = readVocab(lang, dataSet)
    encoder = torch.load(setting.MODEL_HOME + "/%s.%s.encoder.pkl" % (dataSet, lang))
    decoder = torch.load(setting.MODEL_HOME + "/%s.%s.decoder.pkl" % (dataSet, lang))

    output_words, attentions = evaluate(nlVocab, codeVocab, encoder, decoder, CodeStr)
    print('> Code  Input: ', CodeStr)
    print('< NL generate: ', ' '.join(output_words))
    showAttention(CodeStr, output_words, attentions)

def evaluateAndShowAttention(nlVocab, codeVocab, encoder, attn_decoder,input_sentence):
    output_words, attentions = evaluate(nlVocab, codeVocab, encoder, attn_decoder, input_sentence)
    print('> Code  Input: ', input_sentence)
    print('< NL generate: ', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)