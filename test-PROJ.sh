#!/bin/bash

tgt_lang=$1
src=udt/en/
tgt=udt/$tgt_lang/

f_test=$tgt/$tgt_lang-universal-test-brown.conll
f_output=$f_test.predict

model=$src/model.proj/model
f_conf=conf/nndep.cfg

./nndep -cltest  $f_test \
        -model $model \
        -output $f_output \
        -cfg $f_conf \
        -clemb resources/projected/en-$tgt_lang/$tgt_lang.50

