#!/bin/bash

cd /export/a04/jguo/work/parser/clnndep
corpus=ptb/
#window=$1
window=${window}

f_train=$corpus/train-proj.dep
f_dev=$corpus/dev.dep
f_model=$corpus/model.wmt11-w$window.d50.h400

f_conf=conf/nndep_mono.cfg

./nndep -train $f_train \
        -dev $f_dev     \
        -model $f_model \
        -cfg $f_conf \
        -emb resources/mono/wmt11-w$window.emb

