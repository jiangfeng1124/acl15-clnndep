#include "DependencySent.h"
#include "strutils.h"

using namespace std;

DependencySent::DependencySent()
{
    init();
}

DependencySent::DependencySent(const DependencySent& s)
{
    n = s.n;
    words = s.words;
    poss = s.poss;
    clusters = s.clusters;
    // pposs = s.pposs;
}

void DependencySent::add(string& word, string& pos, string& cluster)
{
    ++ n;
    words.push_back(word);
    poss.push_back(pos);
    clusters.push_back(cluster);
    // pposs.push_back(ppos);
}

void DependencySent::init()
{
    n = 0;
    words.clear();
    poss.clear();
    clusters.clear();
    // pposs.clear();
}

void DependencySent::print_info()
{
    cerr << "n = " << n << endl
         << join(words, ",") << endl
         << join(poss, ",")  << endl
         << join(clusters, ",") << endl;
         // << join(pposs, ",") << endl;
}
