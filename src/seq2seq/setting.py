# -*- coding: utf-8 -*-
# 设置数据集文件目录
BASE_DIR = "/home/zhouql"
# BASE_DIR = "/home/zhou/0-Research"
HOME_DIR = BASE_DIR + "/CodeSumy"
WORKDIR = HOME_DIR + "/data/workdir"
PNG_HOME = HOME_DIR + "/data/pngSaved"

MODEL_HOME = BASE_DIR + "/CodeSumy/src/model/modelSaved"
# 设置数据集的构造 设置数据集划分比例和规模
DOWNLOAD_DATA = False
SPLITE_DATA = True

DATASET_SIZE = 0  # DATASET_SIZE = 0 时,选择data.txt中所有的数据
TRAIN_PROP = 0.8
VALID_PROP = 0.1
TEST_PROP = 0.1

# 构造词表的过滤阈值
JAVA_UNK_THRESHOLD = 2
CSHARP_UNK_THRESHOLD = 2

TEXT_UNK_THRESHOLD = 2

# 是否使用cuda
USE_CUDA = True

# 设置神经网络参数
teacher_forcing_ratio = 0.5
HIDDDEN_SIAZE = 256

# Seq2Seq参数
SOS_TOKEN = 2
EOS_TOKEN = 3
UNK_TOKEN = 0
PAD_TOKEN = 0

# 为了训练速度，太长的句子不进行训练，设定阈值
SENTENCE_MAX_LENGTH = 100
