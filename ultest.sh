#!/bin/bash

tgt_lang=$1
src=udt/en/
#tgt=udt/$tgt_lang/
#src=conllx/en/
tgt=conllx/$tgt_lang/

#f_dev=$tgt/test-proj.dep
f_dev=$tgt/$tgt_lang-universal-test.conll
f_output=$f_dev.predict

sample_dev=ptb/sample2.dep

#model=$src/model.fixemb.cca.d100.h400
#model=ptb/model.pre-trained.autopos-stanford.dropout.h400
#model=conllx/en/model.ul.delexicalized.d100.h400
model=udt/en/model.delexicalized.d100.h400

./nndep -test  $f_dev \
        -model $model \
        -output $f_output \
        -cfg nndep.cfg

