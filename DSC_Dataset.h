#ifndef __NNDEP_DATASET_H__
#define __NNDEP_DATASET_H__

#include <vector>
#include "DSCTree.h"

class Sample
{
    public:
        Sample(const Sample & smp)
        {
            feature = smp.feature;
            label = smp.label;
            tree = smp.tree;
        }

        Sample(std::vector<int> & _feature, DSCTree & _tree, std::vector<int> & _label) : \
            feature(_feature),
            label(_label),
            tree(_tree) {};
        ~Sample();

        std::vector<int> & get_feature();
        std::vector<int> & get_label();
        DSCTree & get_dsctree();

    private:
        std::vector<int> feature;
        std::vector<int> label;
        DSCTree tree;
};

class Dataset
{
    public:
        Dataset() {}
        Dataset(const Dataset & ds)
        {
            n = ds.n;
            num_features = ds.num_features;
            num_labels = ds.num_labels;
            samples = ds.samples;
        }

        Dataset(int num_features, int num_labels) : \
            n(0), \
            num_features(num_features), \
            num_labels(num_labels){}
        ~Dataset();

        void add_sample(std::vector<int> & feature,
                        DSCTree & tree,
                        std::vector<int> & label);
        void print_info();

        void shuffle();

    public:
        int n;
        int num_features;
        int num_labels;
        std::vector<Sample> samples;
};


#endif

