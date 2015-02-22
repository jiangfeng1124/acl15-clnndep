#!/bin/bash

#corpus=../../ptb/conll/corpus/auto-stanford/
tgt_lang=$1
src=udt/en/
tgt=udt/$tgt_lang/

f_dev=$tgt/$tgt_lang-universal-test.conll
f_output=$f_dev.predict

model=$src/model.cca.en-$tgt_lang.clean.d.v.d50.h400/model
# model=$src/model.cca.en-$tgt_lang.d.v.d50.h400
f_conf=conf/nndep_full_dist_valency.cfg

./nndep -cltest  $f_dev \
        -model $model \
        -output $f_output \
        -cfg $f_conf \
        -clemb resources/cca/en-$tgt_lang/$tgt_lang.50.clean

