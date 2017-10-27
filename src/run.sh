#!/bin/bash

# 使用pytorch环境
source activate pytorch

# 准备数据集
python ./buildData/buildData.py 

# 进行模型的训练
python ./model/main.py
