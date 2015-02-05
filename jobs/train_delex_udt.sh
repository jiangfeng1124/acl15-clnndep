#!/bin/bash

#corpus=../../ptb/conll/corpus/auto-stanford/
#lang=$1
workspace=/home/jguo29/work/parser/tparser-1.3/src/nndep/
#corpus=$workspace/stanford/
corpus=$workspace/udt/en/

f_train=$corpus/en-universal-train-proj.conll
#f_train=../../ptb/conll/corpus/autopos-zpar/train-proj.dep
#f_train=../../ptb/conll/corpus/autopos-matt/train.dep
f_dev=$corpus/en-universal-dev.conll
#f_dev=../../ptb/conll/corpus/autopos-zpar/dev-proj.dep
#f_dev=../../ptb/conll/corpus/autopos-matt/dev.dep
#f_model=ptb/model.pre-trained.autopos-stanford.dropout.h400.shuffle
#f_model=ptb/model.fix_emb.bicvm-en-de.h200
#f_model=$corpus/model.finetuned.senna.h200
#f_model=$corpus/model.fixemb.bicvm-en-fr.d50.h200
#f_model=$corpus/model.delexicalized.usedist.d100.h400
#f_model=$corpus/model.fixemb.cca.en-$lang.d100.h400
#f_model=$corpus/model.delexicalized.d100.h400
f_model=$corpus/model.delex.d100.h400

sample_train=ptb/train.dep
sample_dev=ptb/dev2.dep

exe=$workspace/nndep
f_cfg=$workspace/nndep_delex.cfg
f_log=$workspace/udt/en/delex.log

$exe -train $f_train \
     -dev $f_dev     \
     -model $f_model \
     -cfg $f_cfg > $f_log 2>&1

