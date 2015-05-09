#include <iostream>
#include <vector>
#include <algorithm>

#include "Dataset.h"
#include "strutils.h"

using namespace std;

vector<int> & Sample::get_feature()
{
    return feature;
}

vector<int> & Sample::get_label()
{
    return label;
}

Sample::~Sample()
{
//     feature.clear();
}

Dataset::~Dataset()
{
//     samples.clear();
}

void Dataset::add_sample(vector<int> & feature, vector<int> & label)
{
    Sample sample(feature, label);
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

int test_Dataset(int argc, char** argv)
{
    Dataset dataset(4, 2);

    int if1[] = {5, 12, 0, 18};
    int if2[] = {9, 2, 40, 78};
    int il1[] = {-1, 0, 0, 1};
    int il2[] = {0, 0, -1, 1};
    vector<int> f1, l1;
    vector<int> f2, l2;
    f1.assign(if1, if1+4); l1.assign(il1, il1+4);
    f2.assign(if2, if2+4); l2.assign(il2, il2+4);

    dataset.add_sample(f1, l1);
    dataset.add_sample(f2, l2);

    dataset.print_info();

    return 0;
}
