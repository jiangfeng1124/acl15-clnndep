#include "ParsingSystem.h"
#include "strutils.h"

#include <iostream>

using namespace std;

ParsingSystem::~ParsingSystem()
{
}

int ParsingSystem::get_transition_id(const string & s)
{
    for (size_t i = 0; i < transitions.size(); ++i)
        if (transitions[i] == s)
            return i;
    cerr << "unrecognized label: " << s << endl;
    return -1;
}

set<string> ParsingSystem::get_punctuation_tags()
{
    string en_punc_tags[] = {"``", "''", ".", ",", ":"};
    string zh_punc_tags[] = {"PU"}; // modify it according to the treebanks.

    cerr << "Language = " << lang << endl;

    if (lang == "english")
    {
        set<string> punc_tags_set(
                en_punc_tags,
                en_punc_tags +
                    sizeof(en_punc_tags) / sizeof(en_punc_tags[0]));
        return punc_tags_set;
    }
    else if (lang == "chinese")
    {
        set<string> punc_tags_set(
                zh_punc_tags,
                zh_punc_tags +
                    sizeof(zh_punc_tags) / sizeof(zh_punc_tags[0]));
        return punc_tags_set;
    }
    else
        return set<string>();
}

set<string> ParsingSystem::get_conll_sub_obj_relations()
{
    string de_sub_obj_relations[] = {"SB", "OA", "OC", "OP"};
    string es_sub_obj_relations[] = {"SUJ"};

    if (eval_lang == "german")
    {
        set<string> sub_obj_relations_set(
                de_sub_obj_relations,
                de_sub_obj_relations +
                    sizeof(de_sub_obj_relations) / sizeof(de_sub_obj_relations[0]));
        return sub_obj_relations_set;
    }
    if (eval_lang == "spanish")
    {
        set<string> sub_obj_relations_set(
                es_sub_obj_relations,
                es_sub_obj_relations +
                    sizeof(es_sub_obj_relations) / sizeof(es_sub_obj_relations[0]));
        return sub_obj_relations_set;
    }
    else
        return set<string>();
}

set<string> ParsingSystem::get_udt_sub_obj_relations()
{
    string udt_sub_obj_relations[] = {"dobj", "iobj", "adpobj", "csubj",
                                      "csubjpass", "nsubj", "nsubjpass"};
    set<string> sub_obj_relations_set(
            udt_sub_obj_relations,
            udt_sub_obj_relations +
                sizeof(udt_sub_obj_relations) / sizeof(udt_sub_obj_relations[0]));
    return sub_obj_relations_set;
}

void ParsingSystem::set_language(const string & s)
{
    lang = str_tolower(s);
}

void ParsingSystem::evaluate(
        vector<DependencySent>& sents,
        vector<DependencyTree>& pred_trees,
        vector<DependencyTree>& gold_trees,
        map<string, double>& result)
{
    result.clear();

    if (pred_trees.size() != gold_trees.size())
    {
        cerr << "Error: incorrect number of trees"
             << endl;
        return ;
    }

    set<string> punc_tags = get_punctuation_tags();
    set<string> sub_obj_relations;
    if (eval_corpora == "conllx")
        sub_obj_relations = get_conll_sub_obj_relations();
    else
        sub_obj_relations = get_udt_sub_obj_relations();

    int correct_arcs = 0;
    int correct_arcs_wo_punc = 0;
    int correct_heads = 0;
    int correct_heads_wo_punc = 0;

    int correct_heads_sub_obj = 0;
    int sum_arcs_sub_obj = 0;

    int correct_trees = 0;
    int correct_trees_wo_punc = 0;
    int correct_root = 0;

    int sum_arcs = 0;
    int sum_arcs_wo_punc = 0;

    for (size_t i = 0; i < pred_trees.size(); ++i)
    {
        if (sents[i].n != pred_trees[i].n)
        {
            cerr << "Error: Tree "
                 << (i + 1)
                 << " has incorrect number of nodes"
                 << endl;
            return ;
        }

        if (!pred_trees[i].is_tree())
        {
            cerr << "Error: Tree "
                 << (i + 1)
                 << " is illegal"
                 << endl;
            return ;
        }

        int n_correct_head = 0;
        int n_correct_head_wo_punc = 0;
        int non_punc = 0;

        for (int j = 1; j <= pred_trees[i].n; ++j)
        {
            if (pred_trees[i].get_head(j) == gold_trees[i].get_head(j))
            {
                ++ correct_heads;
                ++ n_correct_head;
                if (pred_trees[i].get_label(j) == gold_trees[i].get_label(j))
                    ++ correct_arcs;
            }
            ++ sum_arcs;

            string tag = sents[i].poss[j-1];
            if (punc_tags.find(tag) == punc_tags.end())
            {
                ++ sum_arcs_wo_punc;
                ++ non_punc;
                if (pred_trees[i].get_head(j) == gold_trees[i].get_head(j))
                {
                    ++ correct_heads_wo_punc;
                    ++ n_correct_head_wo_punc;
                    if (pred_trees[i].get_label(j) == gold_trees[i].get_label(j))
                        ++ correct_arcs_wo_punc;
                }
            }

            string gold_rel = gold_trees[i].get_label(j);
            if (sub_obj_relations.find(gold_rel) != sub_obj_relations.end())
            {
                ++ sum_arcs_sub_obj;
                if (pred_trees[i].get_head(j) == gold_trees[i].get_head(j))
                    ++ correct_heads_sub_obj;
            }
        }

        if (n_correct_head == pred_trees[i].n)
            ++ correct_trees;
        if (non_punc == n_correct_head_wo_punc)
            ++ correct_trees_wo_punc;
        if (pred_trees[i].get_root() == gold_trees[i].get_root())
            ++ correct_root;
    }

    result["UAS"] = correct_heads * 100.0 / sum_arcs;
    result["UASwoPunc"] = correct_heads_wo_punc * 100.0 / sum_arcs_wo_punc;
    result["LAS"] = correct_arcs  * 100.0 / sum_arcs;
    result["LASwoPunc"] = correct_arcs_wo_punc  * 100.0 / sum_arcs_wo_punc;

    result["UEM"] = correct_trees * 100.0 / pred_trees.size();
    result["UEMwoPunc"] = correct_trees_wo_punc * 100.0 / pred_trees.size();
    result["ROOT"] = correct_root * 100.0 / pred_trees.size();

    result["SOUAS"] = correct_heads_sub_obj * 100.0 / sum_arcs_sub_obj;
}

double ParsingSystem::get_uas_score(
        vector<DependencySent>& sents,
        vector<DependencyTree>& pred_trees,
        vector<DependencyTree>& gold_trees)
{
    map<string, double> result;
    evaluate(sents, pred_trees, gold_trees, result);

    if ( result.empty() ||
            result.find("UASwoPunc") == result.end())
        return -1.0;
    else
        return result["UASwoPunc"];
}

