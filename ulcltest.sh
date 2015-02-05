#!/bin/bash

#corpus=../../ptb/conll/corpus/auto-stanford/
tgt_lang=$1
#src=udt/en/
#src=conllx/en/
#tgt=udt/$tgt_lang/

src=stanford/
tgt=conllx/$tgt_lang/

#f_dev=$tgt/test-proj.dep
f_dev=$tgt/$tgt_lang-universal-test.conll
f_output=$f_dev.predict

sample_dev=ptb/sample2.dep

#model=$src/model.fixemb.cca.en-$tgt_lang.d100.h400
#model=$src/model.delex.d100.h400
model=$src/model.cca.en-$tgt_lang.d100.h400

# if delexicalized, use test (not cltest)
./nndep -cltest  $f_dev \
        -model $model \
        -output $f_output \
        -cfg nndep.cfg \
        -clemb resources/cca/en-$tgt_lang/$tgt_lang.100

