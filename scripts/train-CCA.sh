#!/bin/bash

cd /export/a04/jguo/work/parser/clnndep

if [ $# -ne 2 ]; then
    echo "Usage: ./train.sh [lang] [cls|0,1]"
    exit -1
fi

lang=${lang}
cls=${cls}
#lang=$1
#cls=$2

corpus=udt/en/
models=models/

f_train=$corpus/en-universal-train-brown.conll
f_dev=$corpus/en-universal-dev-brown.conll

if [ "$cls" = "1" ]; then
    echo "Train CCA+Cluster"
    model_dir=$models/model.cca.cls.en-$lang
    f_conf=conf/cca-dvc.cfg
else
    echo "Train CCA"
    model_dir=$models/model.cca.en-$lang
    f_conf=conf/cca-dv.cfg
fi

if [ ! -d $model_dir ]; then
    mkdir $model_dir
fi

f_model=$model_dir/model

./bin/clnndep -train $f_train \
              -dev $f_dev     \
              -model $f_model \
              -cfg $f_conf    \
              -emb resources/cca/en-$lang/en.50.w2v

