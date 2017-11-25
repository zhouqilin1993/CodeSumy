# -*- coding: utf-8 -*-
# email: zhouqilin@buaa.edu.cn
# update: 2017/11/18
# 根据 train.txt 将 train.txt test.txt valid.txt 依照词表进行 index 表达

import json
import re
from io import open

import torch
from torch.autograd import Variable

import setting


# Lowercase, trim, and remove non-letter characters
def normalizeString(str):
    s = str.strip().decode('utf-8').encode('ascii', 'replace')
    s = s.lower()
    return s

def tokenizeNL(nl_str):
    nl_str = normalizeString(nl_str).strip()
    nl_token = re.findall(r"[\w]+|[^\s\w]", nl_str)
    # nl_token = re.findall(r"[\w]+", nl_str)  # extract Token from NL using regular expression
    return nl_token

def tokenizeCode(code_str, lang):
    code_str = normalizeString(code_str).strip()

    # Coded的Token提取可以使用ANTLR4构造词法分析程序进行处理,此处先使用正则进行提取
    code_token = None
    if lang == "java":
        # code_token = parseJava(code_str)
        # code_token = [re.sub('\s+', ' ', x.strip()) for x in code_str]
        code_token = re.findall(r"[\w]+", code_str)
    elif lang == "csharp":
        # code_token = parseCSharp(code_str)
        # code_token = [re.sub('\s+', ' ', x.strip()) for x in code_str]
        code_token = re.findall(r"[\w]+", code_str)
    else:
        print("ERROR: wrong language in TokenizeCode!")

    return code_token

class VocabSet:
    def __init__(self):
        self.n_words = 3  # Count UNK, TOKEN_START and TOKEN_END
        self.word2index = {"UNK": 0, "PAD": 1, "SOS": 2, "EOS": 3} # PAD 表示低频词语，UNK 表示不在词表里面
        self.index2word = {0: "UNK", 1: "PAD", 2: "SOS", 3: "EOS"}
        self.word2count = {}

    def vocabSplit(self, TokenSet):
        UNK_THRESHOLD = 2
        if TokenSet.type == "NL":
            UNK_THRESHOLD = setting.TEXT_UNK_THRESHOLD
        elif TokenSet.type == "CODE":
            if TokenSet.lang == "java":
                UNK_THRESHOLD = setting.JAVA_UNK_THRESHOLD
            elif TokenSet.lang == "csharp":
                UNK_THRESHOLD = setting.CSHARP_UNK_THRESHOLD
            else:
                print ("Error: language must be java or charp!\n")
        else:
            print ("Error: TokenSet.type must be NL or CODE!\n")

        for key in TokenSet.token2count.keys():
            if TokenSet.token2count[key] <= UNK_THRESHOLD:
                continue
            if key not in self.word2count.keys():
                self.word2index[key] = int(self.n_words)
                self.word2count[key] = int(TokenSet.token2count[key])
                self.index2word[int(self.n_words)] = key
                self.n_words += 1
            else:
                self.word2count[key] = TokenSet.token2count[key]
    def initSet(self, vocabSet):
        self.n_words = int(vocabSet['n_words'])
        self.word2index = vocabSet['word2index']
        self.index2word = vocabSet['index2word']
        self.word2count = vocabSet['word2count']

    def getWord(self, index):
        return self.index2word[str(index)]

    def getIndex(self, word):
        if word in self.word2index.keys():
            index = int(self.word2index[word])
        else:
            index = setting.UNK_TOKEN
        return index

class TokenSet:
    def __init__(self, typeName, lang = "NULL"):
        self.type = typeName         # CODE or NL
        self.lang = lang
        self.token2index = {}
        self.index2token = {}
        self.token2count = {}
        self.n_tokens = 0

    def addSentence(self, sentence):
        tokens = None
        if self.type == "NL":
            tokens = tokenizeNL(sentence)
        elif self.type== "CODE":
            tokens = tokenizeCode(sentence, self.lang)
        else:
            print("ERROR: type must be NL or CODE.")

        for token in tokens:
            self.addToken(token)

    def addToken(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[int(self.n_tokens)] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1

    def saveVocab(self, dataSet,lang):
        vocabSet = VocabSet()
        vocabSet.vocabSplit(self)
        Vocab = {
            "n_words": vocabSet.n_words,
            "word2index": vocabSet.word2index,
            "index2word": vocabSet.index2word,
            "word2count": vocabSet.word2count
        }
        with open(setting.WORKDIR + '/%s.%s.%s.vocab' % (dataSet, lang, self.type), 'w') as f:
            f.write(unicode(json.dumps(Vocab)))

    def initSet(self, tokenSet):
        self.type = tokenSet.type
        self.lang = tokenSet.lang
        self.token2index = tokenSet.token2index
        self.index2token = tokenSet.index2token

        self.token2count = int(tokenSet.token2count)
        self.n_tokens = int(tokenSet.n_tokens)

def genVocab(filename, lang = "java", dataSet = "stackoverflow"): # 只使用 train.txt 生成词表
    lines = open(setting.HOME_DIR + '/data/%s/%s/%s' % (dataSet, lang, filename), encoding='utf-8'). \
        read().strip().split('\n')
    nlSet = TokenSet("NL")
    codeSet = TokenSet("CODE", lang)

    for l in lines:
        line = [item for item in normalizeString(l).split('\t')]
        if len(line) == 3:
            nlSet.addSentence(line[1])
            codeSet.addSentence(line[2])

    nlSet.saveVocab(dataSet,lang)
    codeSet.saveVocab(dataSet,lang)


def indexesFromSentence(Vocab, sentence):
    return [Vocab.getIndex(word) for word in sentence.split(' ')]


def variableFromSentence(Vocab, sentence):
    indexes = indexesFromSentence(Vocab, sentence)
    indexes.append(setting.EOS_TOKEN)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if setting.USE_CUDA:
        return result.cuda()
    else:
        return result

# 导致前期速度慢的主要原因
def variablesPairsFromData(dataType, lang, dataSet):
    nlVocab, codeVocab = readVocab(lang, dataSet)
    lines = open(setting.WORKDIR + '/%s.%s.%s.data' % (dataSet, lang, dataType), encoding='utf-8'). \
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[pair[2],pair[1]] for pair in pairs if len(pair) == 3 and \
             len(pair[2]) < setting.SENTENCE_MAX_LENGTH and len(pair[1]) < setting.SENTENCE_MAX_LENGTH]
    #train_pairs = [[ pair[0], pair[1]] for pair in pairs]
    train_pairs = [[variableFromSentence(codeVocab,pair[0]),variableFromSentence(nlVocab,pair[1])] for pair in pairs]

    return pairs, train_pairs

def readVocab(lang, dataSet):
    nlVocab = VocabSet()
    codeVocab = VocabSet()
    nl_str = open(setting.WORKDIR + '/%s.%s.NL.vocab' % (dataSet, lang)).read().strip()
    code_str = open(setting.WORKDIR + '/%s.%s.CODE.vocab' % (dataSet, lang)).read().strip()
    nlVocab.initSet(json.loads(nl_str))
    codeVocab.initSet(json.loads(code_str))

    return nlVocab,codeVocab

def genDataSet(dataSetType, lang, dataSet):
    lines = open(setting.HOME_DIR + '/data/%s/%s/%s.txt' % (dataSet, lang, dataSetType)).\
        read().strip().split('\n')
    f = open(setting.WORKDIR + "/%s.%s.%s.data" % (dataSet, lang, dataSetType), 'w')
    for line in lines:
        l = [item for item in normalizeString(line).split('\t')]
        if len(l) == 3:
            line_id, line_text, line_code = l
            line_str = line_id + '\t'
            for word in tokenizeNL(line_text):
                line_str = line_str + word + ' '
            line_str = line_str.strip() + '\t'
            for code in tokenizeCode(line_code,lang):
                line_str = line_str + code + ' '
            line_str = line_str.strip() + '\n'
            f.write(unicode(line_str))
    f.close()

if __name__ == '__main__':
    genVocab("train.txt","java","so")
    genVocab("train.txt", "csharp", "so")

    genDataSet("train","java","so")
    genDataSet("test", "java", "so")
    genDataSet("valid", "java", "so")

    genDataSet("train","csharp","so")
    genDataSet("test", "csharp", "so")
    genDataSet("valid", "csharp", "so")





