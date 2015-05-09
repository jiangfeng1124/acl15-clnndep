/**
 * stand-alone script for evaluating UAS/LAS.
 */

#include <iostream>
#include <vector>
#include <map>
#include "ArcStandard.h"
#include "Util.h"

using namespace std;

int main(int argc, char ** argv)
{
    if (argc != 3)
    {
        cerr << argv[0] << " [reference] [predict]" << endl;
        exit(-1);
    }

    string ref = argv[1];
    string prd = argv[2];

    vector<DependencySent> ref_sents;
    vector<DependencyTree> ref_trees;

    vector<DependencySent> prd_sents;
    vector<DependencyTree> prd_trees;

    Util::load_conll_file(ref.c_str(), ref_sents, ref_trees);
    Util::load_conll_file(prd.c_str(), prd_sents, prd_trees);

    map<string, double> result;
    ArcStandard system;
    system.set_language("english");
    system.evaluate(ref_sents, prd_trees, ref_trees, result);

    double uas = result["UASwoPunc"];
    double las = result["LASwoPunc"];
    double uem = result["UEMwoPunc"];
    double root = result["ROOT"];

    cerr << "UAS  = " << uas  << "%" << endl;
    cerr << "LAS  = " << las  << "%" << endl;
    cerr << "UEM  = " << uem  << "%" << endl;
    cerr << "ROOT = " << root << "%" << endl;

    return 0;
}

