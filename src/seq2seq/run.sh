#!/bin/bash

# 使用pytorch环境
source activate pytorch

# 下载数据集
python ./buildData.py
# 生成词表
python ./buildVocab.py

# 训练模型
python ./train.py

# 测试模型
python ./evaluate.py


