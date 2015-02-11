#!/bin/bash

cd /export/a04/jguo/work/parser/clnndep
corpus=ptb/

f_train=$corpus/train-proj.dep
f_dev=$corpus/dev.dep
f_model=$corpus/model.senna.d100.h400

f_conf=conf/nndep_mono.cfg

./nndep -train $f_train \
        -dev $f_dev     \
        -model $f_model \
        -cfg $f_conf \
        -emb resources/senna.emb

