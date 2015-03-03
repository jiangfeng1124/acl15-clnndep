#!/bin/python
# -*- coding: utf-8 -*-

import sys
from editdistance import eval
from collections import defaultdict
import operator
import random

def is_digit1(w):
    return len(w) == 1 and w.isdigit()

def is_digit4(w):
    return len(w) == 4 and w.isdigit()


en_embeddings = {}
fo_embeddings = {}

def load_embedding(path, dic):
    ''' load word embedding matrix
    '''
    for i,l in enumerate(open(path)):
        print >> sys.stderr, "\r%d" % (i),
        l = l.strip().split()
        # dic[l[0]] = map(float, l[1:])
        dic[l[0]] = l[1:]
    print >> sys.stderr

if len(sys.argv) != 4:
    print >> sys.stderr, "%s [depfile] [enemb] [foemb] > output" % (sys.argv[0])
    exit(1)

en_emb_file = sys.argv[2]
fo_emb_file = sys.argv[3]

dep = sys.argv[1]

print >> sys.stderr, "load en embeddings"
load_embedding(en_emb_file, en_embeddings)
print >> sys.stderr, "load fr embeddings"
load_embedding(fo_emb_file, fo_embeddings)

threshold = 1

def find_most_similar(w, thresh):
    ed = defaultdict(list)
    for wd,e in fo_embeddings.items():
        # if w[:2] != wd[:2]: continue # prefix constraint
        d = eval(w, wd)
        if d > thresh: continue
        ed[d].append(wd)
    if len(ed) == 0: return None

    for i in range(1, thresh+1):
        if i in ed:
            return ed[i]

lang = sys.argv[3]
cache = {}
# cache["l'"] = en_embeddings["the"]

for l in open(sys.argv[1]):
    l = l.strip().split()
    if len(l) < 2: continue

    word = l[1].lower()
    if word in cache: continue

    ### add these words to alignment dictionary
    # "l' ||| the", "d' ||| of", "n' ||| n't", "c' ||| it", "j' ||| I"

    # if word == "l'":
    # elif word == "d'":      l[5] = "110100"
    # elif word == "qu'":     l[5] = "1101111101"
    # elif word == "n'":      l[5] = "1111010101"
    # elif word == "c'":      l[5] = "11100111"
    # elif word == "j'":      l[5] = "1110000"
    # elif word == "jusqu'":  l[5] = "11011110011"
    # elif word == "lorsqu'": l[5] = "1101111010110"
    # elif word == "États":   l[5] = "100010011"
    # elif word == "aujourd'":l[5] = "10111001110"

    if word not in fo_embeddings:
        if word == "qu'":
            cache[word] = fo_embeddings["qui"]
        if is_digit1(word):
            cache[word] = en_embeddings["4"]
        elif is_digit4(word):
            cache[word] = en_embeddings["2019"]
        elif word.isdigit():
            cache[word] = en_embeddings["30"]

        elif len(word) <= 2 or len(word) > 5:
            # if len(word) <= 2:
            cands = find_most_similar(word, threshold)
            if cands != None:
                # index = random.randint(0, len(cands)-1)
                # cache[word] = fo_embeddings[cands[index]]

                # cache[word] = average(cands)
                cache[word] = []

                for i,cand in enumerate(cands):
                    cand_emb = fo_embeddings[cand]
                    if i == 0:
                        cache[word] = [float(v) / len(cands) for v in cand_emb]
                    else:
                        for j in xrange(len(cand_emb)):
                            cache[word][j] += float(cand_emb[j]) / len(cands)

                cache[word] = map(str, cache[word])
                print >> sys.stderr, "assign %s -> %s" % (word, ",".join(cands))

for w,e in cache.items():
    print "%s %s" % (w, " ".join(e))
