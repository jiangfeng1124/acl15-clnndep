#!/bin/bash

tgt_lang=$1
src=udt/en/
tgt=udt/$tgt_lang/

#f_dev=$tgt/test-proj.dep
f_dev=$tgt/$tgt_lang-universal-test.conll
f_output=$f_dev.predict

model=$src/model.delex.d100.h400
f_conf=conf/nndep_delex.cfg

./nndep_v1 -test  $f_dev \
        -model $model \
        -output $f_output \
        -cfg $f_conf

