import codecs
import csv
from metrics.metric import metricPair

def createNmtTestPair():
    with codecs.open("./opennmt/data/src-test-so-java.txt", "r", "utf-8") as src_test:
        src_tests = src_test.readlines()
    with codecs.open("./opennmt/data/tgt-test-so-java.txt", "r", "utf-8") as tgt_test:
        tgt_tests = tgt_test.readlines()
    with codecs.open("./opennmt/so-java-pred.txt", "r", "utf-8") as gen_test:
        gen_tests = gen_test.readlines()

    nmt_metrics = []
    # {'Bleu1': BLEU_1, 'Bleu2': BLEU_2, 'Bleu3': BLEU_3, 'Bleu4': BLEU_4, 'Rouge': ROUGE_L}
    # for i in range(len(src_tests)):
    #     metrics = metricPair([str(tgt_tests[i])],[str(gen_tests[i])])
    #     line = str(metrics['Bleu1'])+'\t'+str(metrics['Bleu2'])+'\t'+str(metrics['Bleu3'])+'\t'+\
    #             str(metrics['Bleu4'])+'\t'+str(metrics['Bleu1'])+'\t'+gen_tests[i]+'\t'+tgt_tests[i]+\
    #             '\t'+src_tests[i]+'\n'
    #     nmt_metrics.append(line)
    #
    # with codecs.open("opennmt-metrics-so-java.txt", "w", "utf-8") as f:
    #     f.writelines(nmt_metrics)

    for i in range(len(src_tests)):
        metrics = metricPair([str(tgt_tests[i])],[str(gen_tests[i])])
        line = [str(metrics['Bleu1']),str(metrics['Bleu2']),str(metrics['Bleu3']),str(metrics['Bleu4']),\
                str(metrics['Bleu1']),gen_tests[i],tgt_tests[i],src_tests[i]]
        nmt_metrics.append(line)
    with open("opennmt-metrics-so-java.csv", "w") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['Bleu1','Bleu2','Bleu3','Bleu4','Rouge','NL_GEN','NL_TGT','CODE_SRC'])
        for line in nmt_metrics:
            writer.writerow(line)


if __name__ == "__main__":
    createNmtTestPair()