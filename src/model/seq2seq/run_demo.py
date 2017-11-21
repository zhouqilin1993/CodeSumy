# -*- coding: utf-8 -*-


import setting
import torch
from utils import evalDemo

encoder = torch.load(setting.MODEL_HOME+"/encoder.pkl")
attn_decoder = torch.load(setting.MODEL_HOME+"/decoder.pkl")

input = "Stop!"

evaluateAndShowAttention(encoder,attn_decoder,input)