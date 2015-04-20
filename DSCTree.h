#ifndef __NNDEP_DSC_TREE_H__
#define __NNDEP_DSC_TREE_H__

#include <unordered_map>
#include <vector>

// typedef unordered_map map

class DSCTree
{
    public:
        DSCTree();
        DSCTree(const DSCTree & dt)
        {
            tree = dt.tree;
            relations = dt.relations;
            wordids = dt.wordids;
            indegree = dt.indegree;
        }
        DSCTree & operator=(const DSCTree & dt)
        {
            tree = dt.tree;
            relations = dt.relations;
            wordids = dt.wordids;
            indegree = dt.indegree;

            return *this;
        }
        void add_arc(int h, int m, int r);
        void add_word(int h, int h_wid);

    public:
        std::unordered_map<int, std::vector<int> > tree;
        std::unordered_map<int, int> wordids;
        std::unordered_map<int, int> relations;
        std::unordered_map<int, int> indegree;
};

#endif
