#!/bin/bash

corpus=../../ptb/conll/corpus/autopos-stanford/
tgt_lang=$1
src=udt/en/
#tgt=udt/$tgt_lang/

f_dev=$corpus/test.dep
#f_dev=$tgt/test-proj.dep
#f_dev=$tgt/$tgt_lang-universal-test.conll
f_output=$f_dev.predict

sample_dev=ptb/sample2.dep

#model=$src/model.fixemb.cca.d100.h400
#model=ptb/model.pre-trained.autopos-stanford.dropout.h400
model=$src/model.proj.d.v.c10.t74.d50.h400

./nndep -test  $f_dev \
        -model $model \
        -output $f_output \
        -cfg nndep_mono.cfg

