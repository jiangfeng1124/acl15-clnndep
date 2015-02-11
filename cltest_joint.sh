#!/bin/bash

#corpus=../../ptb/conll/corpus/auto-stanford/
tgt_lang=$1
src=udt/en/
tgt=udt/$tgt_lang/

f_dev=$tgt/$tgt_lang-universal-test.conll
f_output=$f_dev.predict

model=$src/model.joint-10.en-$tgt_lang.d100.h400
f_conf=conf/nndep_full.cfg

./nndep -cltest  $f_dev \
        -model $model \
        -output $f_output \
        -cfg $f_conf \
        -clemb resources/joint-10/en-$tgt_lang/$tgt_lang.100

