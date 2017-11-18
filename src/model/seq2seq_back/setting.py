# -*- coding: utf-8 -*-
# 设置数据集文件目录
HOME_DIR = "/home/zhouql/CodeSumy"
WORKDIR = HOME_DIR + "/data/workdir"
PNG_HOME = HOME_DIR + "/data/pngSaved"

MODEL_HOME = HOME_DIR + "/src/model/model_saved"
# 设置数据集的构造
DOWNLOAD_DATA = False
SPLITE_DATA = False
# 设置数据集划分比例和规模
DATASET_SIZE = 200  # DATASET_SIZE = 0 时,选择data.txt中所有的数据

TRAIN_PROP = 0.8
VALID_PROP = 0.1
TEST_PROP = 0.1

# 构造词表的过滤阈值
CODE_UNK_THRESHOLD = 1
TEXT_UNK_THRESHOLD = 1

# 是否使用cuda
USE_CUDA = True

# 设置神经网络参数
teacher_forcing_ratio = 0.5

# Seq2Seq参数
SOS_token = 0
EOS_token = 1
# 为了训练速度，太长的句子不进行训练，设定阈值
SENTENCE_MAX_LENGTH = 50

# Seq2Seq nmt 的前缀处理
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)