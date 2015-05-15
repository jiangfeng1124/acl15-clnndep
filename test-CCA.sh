#!/bin/bash

tgt_lang=$1
src=udt/en/
tgt=udt/$tgt_lang/

f_test=$tgt/$tgt_lang-universal-test-brown.conll
f_output=$f_test.predict

#model=$src/model.cca.en-$tgt_lang/model
model=$src/model.cca.en-$tgt_lang.clean.w2v.d.v.c8.d50.h400/model
f_conf=conf/nndep.cfg

./bin/clnndep -cltest  $f_test \
              -model $model \
              -output $f_output \
              -cfg $f_conf \
              -clemb resources/cca/en-$tgt_lang/$tgt_lang.50.w2v

