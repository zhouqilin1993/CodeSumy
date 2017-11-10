#!/bin/bash

# 使用pytorch环境
source activate pytorch

# 下载数据集
python ./buildData/getData.py
# 数据集预处理
python ./buildData/buildData.py

# 进行模型的训练
python ./model/main.py
