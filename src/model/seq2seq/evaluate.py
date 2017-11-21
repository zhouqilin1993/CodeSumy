# -*- coding: utf-8 -*-

import random

import torch
from torch.autograd import Variable

import setting
from buildVocab import variableFromSentence, readVocab

def evaluate(nlVocab, codeVocab, encoder, decoder, sentence, max_length=setting.SENTENCE_MAX_LENGTH):
    input_variable = variableFromSentence(codeVocab, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if setting.USE_CUDA else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[setting.SOS_TOKEN]]))  # SOS
    decoder_input = decoder_input.cuda() if setting.USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == setting.EOS_TOKEN:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(nlVocab.getIndex(ni))

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if setting.USE_CUDA else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(lang, dataSet, encoder, decoder, pairs,n=10):
    nlVocab, codeVocab = readVocab(lang, dataSet)
    for i in range(n):
        pair = random.choice([pair for pair in pairs if len(pair[0].split(' '))<setting.SENTENCE_MAX_LENGTH and \
                              len(pair[1].split(' '))<setting.SENTENCE_MAX_LENGTH])
        print('> CodeInput:', pair[0])
        print('= NL target:', pair[1])
        output_words, attentions = evaluate(nlVocab, codeVocab, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('NL generate:', output_sentence)
        print('')