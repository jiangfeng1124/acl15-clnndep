#!/bin/bash

# lang=${lang}
corpus=udt/en/

f_train=$corpus/en-universal-train-brown.conll
f_dev=$corpus/en-universal-dev-brown.conll

model_dir=$corpus/model.proj

if [ ! -d $model_dir ]; then
    mkdir $model_dir
fi
f_model=$model_dir/model
f_conf=conf/nndep.cfg

./bin/clnndep -train $f_train \
              -dev $f_dev     \
              -model $f_model \
              -cfg $f_conf    \
              -emb resources/projected/en/en.50

