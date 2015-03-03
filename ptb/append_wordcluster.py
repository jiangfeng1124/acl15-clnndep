#!/bin/python

import sys

fin = sys.argv[1]
fcluster = sys.argv[2]

__UNKNOWN__ = "-UNKNOWN-"

clusters = {}
def load_brown_cluster(path):
    for l in open(path):
        l = l.strip().split()
        clusters[l[1]] = l[0]

load_brown_cluster(fcluster)

for l in open(fin):
    l = l.strip().split()
    if len(l) < 10:
        print
        continue
    word = l[1].lower()
    if word in clusters:
        l[5] = clusters[word]
    else:
        l[5] = __UNKNOWN__
    print "\t".join(l)

