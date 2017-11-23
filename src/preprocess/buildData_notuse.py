# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from src.model.seq2seq import setting
import re
import collections
import json

# 将下载得到的数据进行预处理,处理结果放置到到workdir目录中

def tokenizeNL(nl):
    nl = nl.strip().decode('utf-8').encode('ascii', 'replace')
    return re.findall(r"[\w]+|[^\s\w]", nl)

def tokenizeCode(code, lang):
    code = code.strip().decode('utf-8').encode('ascii', 'replace')

    # Coded的Token提取可以使用ANTLR4构造词法分析程序进行处理,此处先使用正则进行提取
    # typedCode = None
    # if lang == "java":
    #   typedCode = parseJava(code)
    # elif lang == "csharp":
    #   typedCode = parseCSharp(code)

    # tokens = [re.sub( '\s+', ' ', x.strip())  for x in typedCode]

    tokens = re.findall(r"[\w]+|[^\s\w]", code)
    return tokens

def buildVocab(plat,lang):
    filename = setting.HOME_DIR + "/data/" + plat + "/" + lang + "/data.txt"
    words = collections.Counter()
    tokens = collections.Counter()
    for line in open(filename, "r"):
        Lid, Lnl, Lcode = line.strip().split('\t')
        tokens.update(tokenizeCode(Lcode, lang))
        words.update(tokenizeNL(Lnl))
    
    fa = open(setting.WORKDIR + '/vocab.'+lang+'.text', 'w')
    fb = open(setting.WORKDIR + '/vocab.'+lang+'.code', 'w')
    for tok in tokens:
        if tokens[tok] > setting.CODE_UNK_THRESHOLD:
            fb.write(tok + '\t' + str(tokens[tok]) + '\n')

    for wd in words:
        if words[wd] > setting.TEXT_UNK_THRESHOLD:
            fa.write(wd + '\t' + str(words[wd]) + '\n')
    fa.close()
    fb.close()
    f1 = open(setting.WORKDIR + '/' + plat + '.' + lang + '.vocab.text', 'w')
    f1.write(json.dumps(words))
    f1.close()
    f2 = open(setting.WORKDIR + '/' + plat + '.' + lang + '.vocab.code', 'w')
    f2.write(json.dumps(tokens))
    f2.close()

    return


# 获取GitHub和StackOverflow的数据,并将处理后的数据放到workdir目录下
if __name__ == '__main__':
    buildVocab("so","java")
    buildVocab("so","csharp")
    