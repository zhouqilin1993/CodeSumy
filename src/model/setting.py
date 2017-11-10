# -*- coding: utf-8 -*-
# 设置数据集文件目录
HOME_DIR = "/home/zhou/0-Research/CodeSumy"
WORKDIR = HOME_DIR + "/data/workdir"
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

# 设置神经网络参数
