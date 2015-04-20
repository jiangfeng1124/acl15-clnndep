#include "DSCTree.h"
#include <cassert>

using namespace std;

DSCTree::DSCTree()
{
}

void DSCTree::add_arc(int h, int m, int r)
{
    if (tree.find(h) != tree.end())
    {
        tree[h].push_back(m);
    }
    else
    {
        vector<int> child(1, m);
        tree[h] = child;
    }
    relations[m] = r;

    // increse indegrees
    // if (indegree.find(h) == indegree.end()) indegree[h] = 1;
    // if (indegree.find(m) == indegree.end()) indegree[m] = 1;
    assert(indegree.find(h) != indegree.end());
    assert(indegree.find(m) != indegree.end());

    indegree[h] += indegree[m];
}

void DSCTree::add_word(int idx, int wid)
{
    wordids[idx] = wid;
    indegree[idx] = 1;
}

