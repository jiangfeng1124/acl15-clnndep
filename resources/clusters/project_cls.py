#!/bin/python

import sys
from collections import defaultdict

def load_cluster(path):
    ''' load brown cluster file
    '''
    source_lang_dict = {}
    for l in open(path):
        l = l.strip().split()
        source_lang_dict[l[1]] = l[0]
    return source_lang_dict

def load_alignment(path):
    ''' load align dict
    '''
    align_dict = {}
    for l in open(path):
        trans_freq_dict = {}
        l = l.strip().split(" ||| ")
        word = l[0]
        trans = l[1].split(" ")
        for e in trans:
            e_word, e_freq = e.split("__")
            trans_freq_dict[e_word] = int(e_freq)
        align_dict[word] = trans_freq_dict
    return align_dict

def load_punc(path):
    punc_dict = []
    for l in open(path):
        l = l.strip().split()
        punc_dict.append(l[1])
    return punc_dict

if __name__ == '__main__':
    ''' given a clustering dict of source language
        project it to a foreign language, according to an alignment file
    '''

    if len(sys.argv) != 4:
        print >> sys.stderr, \
            "python %s [source_lang_cluster] [alignment] [target_lang_projected_cluster]" % (sys.argv[0])
        exit(1)

    source_cluster_path = sys.argv[1]
    alignment_dict_path = sys.argv[2]
    target_projected_cluster_path = sys.argv[3]
    output = open(target_projected_cluster_path, "w")

    target_projected_cluster_dict = {}
    print >> sys.stderr, "Load source clusters...",
    source_lang_dict = load_cluster(source_cluster_path)
    print >> sys.stderr, "done."
    print >> sys.stderr, "Load alignment dict...",
    align_dict = load_alignment(alignment_dict_path)
    print >> sys.stderr, "done."

    print >> sys.stderr, "Load punc list...",
    punc_dict = load_punc("./punc.lst")
    print >> sys.stderr, "done."

    for w, trans_freq_dict in align_dict.items():
        cluster_freq_dict = defaultdict(int)
        for trans, freq in trans_freq_dict.items():
            if trans in source_lang_dict:
                cluster_freq_dict[source_lang_dict[trans]] += freq
        if len(cluster_freq_dict) == 0:
            continue

        max_freq = 0
        opt_cluster = ""
        for cluster, freq in cluster_freq_dict.items():
            if freq > max_freq:
                opt_cluster = cluster
                max_freq = freq

        target_projected_cluster_dict[w] = opt_cluster

    for w in punc_dict:
        if w in source_lang_dict:
            target_projected_cluster_dict[w] = source_lang_dict[w]

    for w, c in target_projected_cluster_dict.items():
        print >> output, "%s\t%s\t%d" % (c, w, 1)
    output.close()

