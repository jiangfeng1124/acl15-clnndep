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
            roots = dt.roots;
        }
        DSCTree & operator=(const DSCTree & dt)
        {
            tree = dt.tree;
            relations = dt.relations;
            wordids = dt.wordids;
            indegree = dt.indegree;
            roots = dt.roots;

            return *this;
        }
        void add_arc(int h, int m, int r);
        void add_word(int h, int h_wid);
        void add_root(int idx);
        void del_root(int i);
        int get_root(int i);
        void trim(int max_depth);

        void print();

    private:
        void trim_tree(int root, int cur_depth, int max_depth);
        void cut(int root);
        void cal_indegree(int root);

    public:
        std::unordered_map<int, std::vector<int> > tree;
        std::vector<int> roots;
        std::unordered_map<int, int> wordids;
        std::unordered_map<int, int> relations;
        std::unordered_map<int, int> indegree;
};

#endif
