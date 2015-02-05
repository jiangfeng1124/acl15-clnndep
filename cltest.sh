#!/bin/bash

#corpus=../../ptb/conll/corpus/auto-stanford/
tgt_lang=$1
src=udt/en/
tgt=udt/$tgt_lang/

#f_dev=$tgt/test-proj.dep
f_dev=$tgt/$tgt_lang-universal-dev.conll
f_output=$f_dev.predict

sample_dev=ptb/sample2.dep

model=$src/model.fixemb.cca.en-$tgt_lang.d100.h400
delexicalized_model=$src/model.delexicalized.d100.h400

./nndep -test  $f_dev \
        -model $delexicalized_model \
        -output $f_output \
        -cfg nndep.cfg
        # -clemb resources/cca/en-$tgt_lang/$tgt_lang.100

