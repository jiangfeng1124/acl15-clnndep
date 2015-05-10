#include <iostream>
#include <vector>
#include <algorithm>

#include "DSC_Dataset.h"
#include "strutils.h"
#include "DSCTree.h"

using namespace std;

vector<int> & Sample::get_feature()
{
    return feature;
}

vector<int> & Sample::get_label()
{
    return label;
}

DSCTree & Sample::get_dsctree()
{
    return tree;
}

Sample::~Sample()
{
//     feature.clear();
}

Dataset::~Dataset()
{
//     samples.clear();
}

void Dataset::add_sample(vector<int> & feature, DSCTree & tree, vector<int> & label)
{
    Sample sample(feature, tree, label);
    n += 1;
    samples.push_back(sample);
}

void Dataset::print_info()
{
    cout << n << endl;
    for (int i = 0; i < n; ++i)
    {
        cout << join(samples[i].get_feature(), " ") << endl;
        cout << join(samples[i].get_label(), " ") << endl;
    }
}

void Dataset::shuffle()
{
    random_shuffle(samples.begin(), samples.end());
}

