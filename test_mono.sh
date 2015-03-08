#!/bin/bash

corpus=ptb/

f_test=$corpus/dev.dep
f_model=$corpus/model.wmt11.d100.h400/model

f_conf=conf/nndep_mono.cfg

./nndep_eigen -test $f_test \
        -model $f_model \
        -output $f_test.predict \
        -cfg $f_conf \
        -emb resources/wmt11-100.emb
        # -emb resources/senna.emb


