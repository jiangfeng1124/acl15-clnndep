#!/bin/bash

cd /export/a04/jguo/work/parser/clnndep
corpus=ptb/
#window=$1
#window=${window}

f_train=$corpus/train-proj.dep
f_dev=$corpus/dev.dep

model_dir=$corpus/model.senna.d50.h400
if [ ! -d $model_dir ]; then
    mkdir $model_dir
fi
f_model=$model_dir/model

f_conf=conf/nndep_mono.cfg

./nndep -train $f_train \
        -dev $f_dev     \
        -model $f_model \
        -cfg $f_conf \
        -emb resources/senna.emb
        # -emb resources/mono/wmt11-w$window.emb

