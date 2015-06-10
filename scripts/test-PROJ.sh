#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: ./train.sh [lang] [cls|0,1]"
    exit -1
fi

lang=$1
cls=$2

src=udt/en/
tgt=udt/$lang/

models=models/

f_test=$tgt/$lang-universal-test-brown-punc-p1.conll
f_output=$f_test.predict

if [ "$cls" = "1" ]; then
    echo "Test PROJ+Cluster"
    model_dir=$models/model.proj.cls
    f_conf=conf/proj-dvc.cfg
    # projected word embeddings in target language
    f_emb=resources/proj-replicate/$lang/$lang.50.proj.cls.pp
else
    echo "Test PROJ"
    model_dir=$models/model.proj
    f_conf=conf/proj-dv.cfg
    # projected word embeddings in target language
    f_emb=resources/proj-replicate/$lang/$lang.50.proj.pp
fi

f_model=$model_dir/model

./bin/clnndep -cltest  $f_test \
              -model $f_model \
              -output $f_output \
              -cfg $f_conf \
              -clemb $f_emb

