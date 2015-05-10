#include "DSCTree.h"
#include <cassert>
#include <iostream>

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

void DSCTree::add_root(int idx)
{
    roots.insert(roots.begin(), idx);
}

void DSCTree::del_root(int i)
{
    roots.erase(roots.begin() + i);
}

int DSCTree::get_root(int i)
{
    if ((unsigned)i > roots.size() - 1)
        return -1;
    else
        return roots[i];
}

void DSCTree::trim(int max_depth)
{
    if (max_depth < 0) return ;

    // reset indegrees
    for (auto iter = indegree.begin(); iter != indegree.end(); ++iter)
        iter->second = 1;

    for (size_t i = 0; i < roots.size(); ++i)
    {
        int cur_depth = 0;
        trim_tree(roots[i], cur_depth, max_depth);
        cal_indegree(roots[i]);
    }
}

void DSCTree::trim_tree(int root, int cur_depth, int max_depth)
{
    auto iter = tree.find(root);
    if (iter == tree.end()) return;

    if (cur_depth >= max_depth)
    {
        cut(root);
        return;
    }
    auto children = iter->second;
    for (size_t i = 0; i < children.size(); ++i)
        trim_tree(children[i], cur_depth + 1, max_depth);
}

void DSCTree::cut(int root)
{
    auto iter = tree.find(root);
    if (iter == tree.end()) return;
    for (size_t i = 0; i < iter->second.size(); ++i)
        cut(iter->second[i]);
    tree.erase(iter);
}

void DSCTree::cal_indegree(int root)
{
    auto iter = tree.find(root);
    if (iter == tree.end()) return ;

    for (size_t i = 0; i < iter->second.size(); ++i)
    {
        int c = iter->second[i];
        cal_indegree(c);
        indegree[root] += indegree[c];
    }
}

void DSCTree::print()
{
    cerr << "Tree Structure:" << endl;
    for (auto iter = tree.begin(); iter != tree.end(); ++iter)
    {
        cerr << iter->first << "->(";
        size_t i = 0;
        for (; i < iter->second.size() - 1; ++i)
            cerr << iter->second[i] << ",";
        cerr << iter->second[i] << ")" << endl;
    }
    cerr << "Wordids:" << endl;
    for (auto iter = wordids.begin(); iter != wordids.end(); ++iter)
        cerr << iter->first << ": " << iter->second << endl;

    cerr << "Relations:" << endl;
    for (auto iter = relations.begin(); iter != relations.end(); ++iter)
        cerr << iter->first << ": " << iter->second << endl;

    cerr << "Roots:" << endl;
    for (auto iter = roots.begin(); iter != roots.end(); ++iter)
        cerr << *iter << " ";
    cerr << endl;

    cerr << "Indegrees:" << endl;
    for (auto iter = indegree.begin(); iter != indegree.end(); ++iter)
        cerr << iter->first << ": " << iter->second << endl;
}

int unit_test(int argc, char** argv)
{
    DSCTree tree;

    tree.add_word(-1, 0);
    tree.add_word(0, 1); tree.add_root(0);
    tree.add_word(1, 2); tree.add_root(1);
    tree.add_arc(0, 1, 1); tree.del_root(0);
    for (int i = 2; i < 10; ++i)
    {
        tree.add_word(i, i+1);
        tree.add_root(i);
    }
    tree.add_arc(9,  8, 1); tree.del_root(1);
    tree.add_arc(7,  9, 1); tree.del_root(0);
    tree.add_arc(7,  6, 1); tree.del_root(1);
    tree.add_arc(5,  7, 1); tree.del_root(0);
    tree.add_arc(4,  5, 1); tree.del_root(0);
    tree.add_arc(4,  3, 1); tree.del_root(1);
    tree.add_arc(4,  2, 1); tree.del_root(1);

    tree.print();
    tree.trim(2);
    tree.print();

    return 0;
}

