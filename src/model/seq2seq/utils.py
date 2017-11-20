# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
import setting
from evaluate import evaluate


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
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showAttention(attentions,input_sentence, output_words ):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
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


def evaluateAndShowAttention(nlVocab, codeVocab, encoder, attn_decoder,input_sentence):
    output_words, attentions = evaluate(nlVocab, codeVocab, encoder, attn_decoder, input_sentence)
    print('> Code  Input: ', input_sentence)
    print('< NL generate: ', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)