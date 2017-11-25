import codecs

def getPair(lines):
    nl = []
    code = []
    for line in lines:
        l = [ls for ls in line.strip().split('\t')]
        if len(l) == 3:
            nl.append(l[1])
            code.append(l[2])
    nl = [s+'\n' for s in nl]
    code = [s+'\n' for s in code]
    return nl,code

def prepaireData():
    data_set_dir = "/home/zhouql/CodeSumy/data/so/java"
    data_dir = "/home/zhouql/CodeSumy/src/opennmt/data"

    with codecs.open(data_set_dir + "/train.txt", "r", "utf-8") as train_file:
        train_lines = train_file.readlines()
    with codecs.open(data_set_dir + "/valid.txt", "r", "utf-8") as valid_file:
        valid_lines = valid_file.readlines()
    with codecs.open(data_set_dir + "/test.txt", "r", "utf-8") as test_file:
        test_lines = test_file.readlines()

    nl1, code1 = getPair(train_lines)
    nl2, code2 = getPair(valid_lines)
    nl3, code3 = getPair(test_lines)

    with codecs.open(data_dir + "/src-train-so-java.txt", "w", "utf-8") as train_f:
        train_f.writelines(code1)
    with codecs.open(data_dir + "/src-val-so-java.txt", "w", "utf-8") as valid_f:
        valid_f.writelines(code2)
    with codecs.open(data_dir + "/src-test-so-java.txt", "w", "utf-8") as test_f:
        test_f.writelines(code3)

    with codecs.open(data_dir + "/tgt-train-so-java.txt", "w", "utf-8") as train_f2:
        train_f2.writelines(nl1)
    with codecs.open(data_dir + "/tgt-val-so-java.txt", "w", "utf-8") as valid_f2:
        valid_f2.writelines(nl2)
    with codecs.open(data_dir + "/tgt-test-so-java.txt", "w", "utf-8") as test_f2:
        test_f2.writelines(nl3)


if __name__ == "__main__":
    prepaireData()