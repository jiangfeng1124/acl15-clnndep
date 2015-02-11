#!/bin/bash

corpus=ptb/

f_test=$corpus/dev.dep
f_model=$corpus/model.senna.d100.h400

f_conf=conf/nndep_mono.cfg

./nndep -test $f_test \
        -model $f_model \
        -output $f_test.predict \
        -cfg $f_conf \
        -emb resources/senna.emb

