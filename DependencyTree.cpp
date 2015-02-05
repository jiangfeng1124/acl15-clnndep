#include "DependencyTree.h"
#include "Config.h"

#include <iostream>

using namespace std;

DependencyTree::DependencyTree()
{
    init();
}

void DependencyTree::init()
{
    n = 0;
    heads.clear();
    labels.clear();

    heads.push_back(Config::NONEXIST);
    labels.push_back(Config::UNKNOWN);
}

DependencyTree::DependencyTree(const DependencyTree& tree)
{
    n = tree.n;
    heads = tree.heads;
    labels = tree.labels;
}

void DependencyTree::add(int h, const string & l)
{
    ++n;
    heads.push_back(h);
    labels.push_back(l);
}

void DependencyTree::set(int k, int h, const string & l)
{
    heads[k] = h;
    labels[k] = l;
}

int DependencyTree::get_head(int k)
{
    if (k <= 0 || k > n)
        return Config::NONEXIST;
    return heads[k];
}

const string & DependencyTree::get_label(int k)
{
    if (k <= 0 || k > n)
        return Config::NIL;
    return labels[k];
}

int DependencyTree::get_root()
{
    for (int k = 1; k <= n; ++k)
        if (get_head(k) == 0)
            return k;
    return 0; // non tree
}

bool DependencyTree::is_single_root()
{
    int roots = 0;
    for (int k = 1; k <= n; ++k)
        if (get_head(k) == 0)
            roots += 1;
    return (roots == 1);
}

/**
 * Checking if the tree is legal, O(n)
 */
bool DependencyTree::is_tree()
{
    vector<int> h;
    h.push_back(-1);
    for (int k = 1; k <= n; ++k)
    {
        if (get_head(k) < 0 || get_head(k) > n)
            return false;
        h.push_back(-1);
    }
    for (int k = 1; k <= n; ++k)
    {
        int i = k;
        while (i > 0)
        {
            if (h[i] >= 0 && h[i] < k)
                break;
            if (h[i] == k)
                return false;
            h[i] = k;
            i = get_head(i);
        }
    }
    return true;
}

bool DependencyTree::is_projective()
{
    if (!is_tree())
        return false;

    counter = -1;
    return visit_tree(0);
}

bool DependencyTree::visit_tree(int w)
{
    for (int k = 1; k < w; ++k)
        if (get_head(k) == w && visit_tree(k) == false)
            return false;
    counter += 1;
    if (w != counter)
        return false;
    for (int k = w + 1; k <= n; ++k)
        if (get_head(k) == w && visit_tree(k) == false)
            return false;
    return true;
}

void DependencyTree::print()
{
    for (int i = 0; i <= n; ++i)
        cerr << i << " " << get_head(i)
             << ", " << " " << get_label(i);
    cerr << endl;
}

