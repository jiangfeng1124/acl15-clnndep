#!/bin/bash
# -*- coding: utf-8 -*-

import sys
from editdistance import eval
from collections import defaultdict
import operator

def is_digit1(w):
    return len(w) == 1 and w.isdigit()

def is_digit4(w):
    return len(w) == 4 and w.isdigit()

en_clusters = {}
fo_clusters = {}

def load_clusters(path, clusters):
    for l in open(path):
        l = l.strip().split()
        clusters[l[1]] = l[0]

if len(sys.argv) != 4:
    print >> sys.stderr, "%s [depfile] [encluster] [focluster]" % (sys.argv[0])
    exit(1)

en_cluster_file = sys.argv[2]
fo_cluster_file = sys.argv[3]

dep = sys.argv[1]

print >> sys.stderr, "load en clusters"
load_clusters(en_cluster_file, en_clusters)
print >> sys.stderr, "load fo clusters"
load_clusters(fo_cluster_file, fo_clusters)

threshold = 1

def find_most_similar(w, thresh):
    ed = defaultdict(list)
    for wd,c in fo_clusters.items():
        if abs(len(wd) - len(w)) > thresh:
            continue
        d = eval(w, wd)
        if d > thresh: continue
        ed[d].append(wd)
    if len(ed) == 0: return None

    for i in range(1, thresh+1):
        if i in ed:
            return ed[i]

cache = {}
for i, l in enumerate(open(sys.argv[1])):

    if i % 100 == 0:
        print >> sys.stderr, "[%d]" % (i)

    l = l.strip().split()
    if len(l) < 2: continue

    word = l[1].lower()

    # elif word == "''":          l[5] = "1101101"
    # elif word == "000":         l[5] = "101001111"

    if word in cache:   continue; # l[5] = cache[word]

    ### add these words to alignment dictionary
    # elif word == "''":  cache[word] = en_clusters["''"]
    # elif word == "000": cache[word] = en_clusters["000"]

    if word not in fo_clusters:
        if is_digit1(word):
            cache[word] = en_clusters["4"]     # l[5] = "101111111"
        elif is_digit4(word):
            cache[word] = en_clusters["2019"]  # l[5] = "101110111"
        elif word.isdigit():
            cache[word] = en_clusters["30"]    # l[5] = "101111110"

        elif len(word) <= 2 or len(word) > 5:
            # if len(word) <= 2:
            cands = find_most_similar(word, threshold)

            if cands != None:
                sim_cls = defaultdict(int)
                for w in cands:
                    c = fo_clusters[w]
                    sim_cls[c] += 1
                sorted_sim_cls = sorted(sim_cls.items(), key = operator.itemgetter(1), reverse=True)
                cache[word] = sorted_sim_cls[0][0]
                print >> sys.stderr, "assign %s -> %s" % (word, ",".join(cands))

    # print "\t".join(l)

for w, c in cache.items():
    print "%s\t%s\t1" % (c, w)

