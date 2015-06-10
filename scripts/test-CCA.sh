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
    echo "Test CCA+Cluster"
    model_dir=$models/model.cca.cls.en-$lang
    f_conf=conf/cca-dvc.cfg
else
    echo "Test CCA"
    model_dir=$models/model.cca.en-$lang
    f_conf=conf/cca-dv.cfg
fi

f_model=$model_dir/model

./bin/clnndep -cltest $f_test \
              -model $f_model \
              -output $f_output \
              -cfg $f_conf \
              -clemb resources/cca/en-$lang/$lang.50.w2v

