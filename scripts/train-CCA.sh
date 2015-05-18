#!/bin/bash

#cd /export/a04/jguo/work/parser/clnndep

#lang=${lang}
lang=$1
corpus=udt/en/

f_train=$corpus/en-universal-train-brown.conll
f_dev=$corpus/en-universal-dev-brown.conll

model_dir=$corpus/model.cca.en-$lang
if [ ! -d $model_dir ]; then
    mkdir $model_dir
fi

f_model=$model_dir/model
f_conf=conf/nndep.cfg

./bin/clnndep -train $f_train \
              -dev $f_dev     \
              -model $f_model \
              -cfg $f_conf    \
              -emb resources/cca/en-$lang/en.50.w2v

