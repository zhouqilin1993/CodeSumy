# -*- coding: utf-8 -*-
import random
import re
import unicodedata

from pymongo import *

import setting


# 从MongoDB中获取Java和C#的数据,提取NL和Code的Token,过滤所有\t的字符(将\t替换为\space);
# 生成的数据集每一行表示一个数据Code NL,分为 data.txt train.txt(80%) test.txt(10%) valid.txt(10%),用\t进行划分
# 将数据过滤停用词等,构造为Token,将清理后的数据放到workdir目录下,命名为
#      (github|stackoverflow).(java|csharp).(data|train|test|valid|vocab)

def download_data(plat,lang):
    client = MongoClient("192.168.7.113",30000)
    db = client.code2text
    collection = db[plat+"_"+lang]

    dataset = []
    id = 0
    for doc in collection.find():
        text = doc["title"].replace('\r','') # MongoDB的字段
        code = doc["code"].replace('\r','')
        docEntry = {"id": id, "text": text, "code":code }
        if ((len(text.strip()) != 0) & (len(code.strip()) != 0)):
            dataset.append(docEntry)
            id = id + 1
        # 加入下载进度条显示

    f = open(setting.HOME_DIR + '/data/' + plat + '/' + lang + '/' + "data.txt", 'w')
    for dataline in dataset:
        line = ""+str(dataline["id"])+"\t"+dataline["text"].replace('\t','\\t').strip() + \
                "\t"+dataline["code"].replace('\t','\\t').strip()
        f.write(re.sub(r'\n',r'\\n',line.strip()).encode('utf-8')+'\n')
    f.close()
    return
# 以下两个函数将下载的数据集进行切分,分为训练集,测试集,验证集
def unicodeToAscii(s):
    return unicodedata.normalize('NFKD', s.decode('utf-8')).encode('ascii','ignore')

def splitData(plat,lang):
    Data_Dir = setting.HOME_DIR + '/data/' + plat + '/' + lang + '/'
    f1 = open(Data_Dir + "train.txt", 'w')
    f2 = open(Data_Dir + "test.txt", 'w')
    f3 = open(Data_Dir + "valid.txt", 'w')

    lines = open(Data_Dir + "data.txt", 'r').read().strip().split('\n')
    if (setting.DATASET_SIZE == 0):
        data_num = len(lines)
    elif(setting.DATASET_SIZE > 0):
        data_num = setting.DATASET_SIZE
    else:
        print("ERROR: DATASET_SIZE MUST >= 0 !")

    data_idx = set(range(data_num))
    train_idx = set(random.sample(data_idx, int(data_num * setting.TRAIN_PROP)))
    valid_idx = set(random.sample(data_idx - train_idx, int(data_num * setting.VALID_PROP)))
    test_idx = set(random.sample(data_idx - train_idx - valid_idx, int(data_num * setting.TEST_PROP)))
    for line in lines:
        l = line.strip().split('\t')
        if len(l) != 3:
            continue
        line_id, line_text, line_code = line.strip().split('\t')
        line_wirte = ""+line_id+"\t"+unicodeToAscii(line_text)+"\t"+unicodeToAscii(line_code)+"\n"
        if int(line_id) in train_idx:
            f1.write(line_wirte)
        elif int(line_id) in valid_idx:
            f3.write(line_wirte)
        elif int(line_id)  in test_idx:
            f2.write(line_wirte)
    f1.close()
    f2.close()
    f3.close()
    return

if __name__ == '__main__':
    # 从MongoDB获取原始数据
    if (setting.DOWNLOAD_DATA):
        download_data("so","java")
        download_data("so","csharp")
    # 将原始数据按照设置切分为Train Test Valid三类数据集
    if (setting.SPLITE_DATA):
        splitData("so","java")
        splitData("so","csharp")
