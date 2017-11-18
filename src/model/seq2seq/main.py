# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import random
import setting
import matplotlib.pyplot as plt
from src.model.seq2seq.buildVocab import prepareData
from evaluate import evaluateRandomly,evaluate
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from train import trainIters
from utils import evaluateAndShowAttention


# Prepare train data
print("Prepare train data...\n")
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

# Training
print("Training...")
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, 1, dropout_p=0.1)

if setting.USE_CUDA:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

# Evaluating
print("Evaluating...")
evaluateRandomly(encoder1, attn_decoder1)

# Visualizing Attention
print("Visualizing Attention...")
output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())

# Some Test Demo
evaluateAndShowAttention(encoder1,attn_decoder1,"elle a cinq ans de moins que moi .")
evaluateAndShowAttention(encoder1,attn_decoder1,"elle est trop petit .")
evaluateAndShowAttention(encoder1,attn_decoder1,"je ne crains pas de mourir .")
evaluateAndShowAttention(encoder1,attn_decoder1,"c est un jeune directeur plein de talent .")

