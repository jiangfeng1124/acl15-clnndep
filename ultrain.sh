#!/bin/bash

#corpus=../../ptb/conll/corpus/auto-stanford/
lang=$1
corpus=conllx/en/

f_train=$corpus/en-universal-train.conll
#f_train=../../ptb/conll/corpus/autopos-zpar/train-proj.dep
#f_train=../../ptb/conll/corpus/autopos-matt/train.dep
f_dev=$corpus/en-universal-test.conll
#f_dev=../../ptb/conll/corpus/autopos-zpar/dev-proj.dep
#f_dev=../../ptb/conll/corpus/autopos-matt/dev.dep
#f_model=ptb/model.pre-trained.autopos-stanford.dropout.h400.shuffle
#f_model=ptb/model.fix_emb.bicvm-en-de.h200
#f_model=$corpus/model.finetuned.senna.h200
#f_model=$corpus/model.fixemb.bicvm-en-fr.d50.h200
#f_model=$corpus/model.delexicalized.usedist.d100.h400
#f_model=$corpus/model.fixemb.cca.en-$lang.d100.h400

# f_model=$corpus/model.ul.delexicalized.d100.h400
f_model=$corpus/model.ul.en-$lang.d100.h400

sample_train=ptb/train.dep
sample_dev=ptb/dev2.dep

./nndep -train $f_train \
        -dev $f_dev     \
        -model $f_model \
        -cfg nndep_ul.cfg \
        -emb resources/cca/en-$lang/en.100

