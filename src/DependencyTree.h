#ifndef __NNDEP_DEPENDENCY_TREE_H__
#define __NNDEP_DEPENDENCY_TREE_H__

#include <vector>
#include <string>

class DependencyTree
{
    public:
        DependencyTree();
        DependencyTree(const DependencyTree& tree);
        ~DependencyTree() {}

        void init();

        void add(int h, const std::string & l);
        void set(int k, int h, const std::string & l);

        int get_head(int k);
        const std::string & get_label(int k);
        int get_root();
        bool is_single_root();

        bool is_tree();
        bool is_projective();

        void print();

    private:
        bool visit_tree(int w);

    public:
        int n;
        std::vector<int> heads;
        std::vector<std::string> labels;
        int counter; // for detecting projectivity
};

#endif
