#!/bin/python

import sys
import operator
from collections import defaultdict

def load_embedding(path):
    ''' load word embedding matrix
    '''
    source_lang_dict = {}
    for i,l in enumerate(open(path)):
        print >> sys.stderr, "\r%d" % (i),
        l = l.strip().split()
        source_lang_dict[l[0]] = map(float, l[1:])
    print >> sys.stderr
    return source_lang_dict

def load_alignment(path):
    ''' load align dict
    '''
    align_dict = {}
    for l in open(path):
        trans_freq_dict = {}
        l = l.strip().split(" ||| ")
        word = l[0]; trans = l[1].split(" ")
        for e in trans:
            e_word, e_freq = e.rsplit("__")
            trans_freq_dict[e_word] = int(e_freq)
        align_dict[word] = trans_freq_dict
    return align_dict

def load_punc(path):
    punc_dict = []
    for l in open(path):
        l = l.strip().split()
        punc_dict.append(l[1])
    return punc_dict

def project(emb_dic, align_dic, punc_dic, weighted, output):
    print >> sys.stderr, "Projecting"
    target_lang_emb_dict = {}
    for w, trans_freq_dic in align_dic.items():
        sorted_trans_freq_dic = sorted(trans_freq_dic.items(),
                                       key=operator.itemgetter(1),
                                       reverse=True)
        if not weighted:    # take the embedding of the most likely translation
            trans = sorted_trans_freq_dic[0][0]
            if trans in emb_dic:
                # print >> output, "%s %s" % \
                #          (w, " ".join(map(str, emb_dic[trans])))
                target_lang_emb_dict[w] = emb_dic[trans]
        elif weighted:      # take the weighted average embedding of its translations
            sum_freq = 0
            w_emb_set = []
            for trans,freq in sorted_trans_freq_dic:
                if trans in emb_dic:
                    sum_freq += freq
                    w_emb_set.append((emb_dic[trans], freq))
            if sum_freq == 0: continue
            w_emb = []
            for i, (vec, freq) in enumerate(w_emb_set):
                if i == 0:
                    w_emb = [v * float(freq) / sum_freq for v in vec]
                else:
                    for j in xrange(len(w_emb)):
                        w_emb[j] += vec[j] * float(freq) / sum_freq
            # print >> output, "%s %s" % (w, " ".join(map(str, w_emb)))
            target_lang_emb_dict[w] = w_emb
    for  w in punc_dict:
        if w in emb_dic:
            target_lang_emb_dict[w] = emb_dic[w];

    for w,e in target_lang_emb_dict.items():
        print >> output, "%s %s" % (w, " ".join(map(str, e)))
    output.close()

if __name__ == '__main__':
    ''' given a embedding dict of source language
        project it to a foreign language, according to an alignment file
    '''

    if len(sys.argv) != 4:
        print >> sys.stderr, \
                "Usage: python %s [source_lang] [target_lang] [weighted=0|1]" % (sys.argv[0])
        exit(1)
    weighted = True if sys.argv[3] == "1" else False

    # source_emb_path = "en/wmt11-envectors.txt"
    # source_emb_path = "%s-%s/%s.50" % (sys.argv[1], sys.argv[2], sys.argv[1])
    source_emb_path = "%s/%s.50.proj.dvc.finetuned" % (sys.argv[1], sys.argv[1])
    alignment_dict_path = "align/%s-%s.align" % (sys.argv[2], sys.argv[1])
    target_projected_emb_path = "%s-%s/%s.50.%d.p.proj.dvc.finetuned" % (sys.argv[1], sys.argv[2], sys.argv[2], weighted)
    output = open(target_projected_emb_path, "w")

    print >> sys.stderr, "Load source embeddings..."
    source_lang_dict = load_embedding(source_emb_path)
    print >> sys.stderr, "done."
    print >> sys.stderr, "Load alignment dict..."
    align_dict = load_alignment(alignment_dict_path)
    print >> sys.stderr, "done."

    print >> sys.stderr, "Load punc list...",
    punc_dict = load_punc("./punc.lst")
    print >> sys.stderr, "done."

    project(source_lang_dict, align_dict, punc_dict, weighted, output)

