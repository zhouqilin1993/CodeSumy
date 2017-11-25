python buildSrcTgtData.py

python preprocess.py -train_src data/src-train-so-java.txt -train_tgt data/tgt-train-so-java.txt -valid_src data/src-val-so-java.txt -valid_tgt data/tgt-val-so-java.txt -save_data data/so-java

python train.py -data data/so-java -save_model so-java-model -gpuid 0

python translate.py -model so-java-model_acc_25.27_ppl_129.00_e7.pt -src data/src-test-so-java.txt -output so-java-pred.txt -replace_unk -verbose
