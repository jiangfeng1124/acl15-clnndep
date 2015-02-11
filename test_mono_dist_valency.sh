#!/bin/bash

corpus=ptb/

f_test=$corpus/test.dep
f_model=$corpus/model.senna.dist.valency.d100.h400

f_conf=conf/nndep_mono_dist_valency.cfg

./nndep -test $f_test \
        -model $f_model \
        -output $f_test.predict \
        -cfg $f_conf \
        -emb resources/senna.emb

