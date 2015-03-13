#include "Util.h"

using namespace std;

int main(int argc, char** argv)
{
    string filepath = argv[1];
    vector<DependencySent> sents;
    vector<DependencyTree> trees;

    Util::load_conll_file(filepath.c_str(), sents, trees, true);

    int n_proj = 0;
    for (size_t i = 0; i < trees.size(); ++i)
    {
        if (trees[i].is_projective())
        {
            n_proj ++;
            cerr << "\r" << i;
            for (int j = 0; j < sents[i].n; ++j)
            {
                int id = j + 1;
                string word = sents[i].words[j];
                string pos  = sents[i].poss[j];
                int head = trees[i].get_head(id);
                string label = trees[i].get_label(id);

                cout << id    << "\t"
                     << word  << "\t" << "_" << "\t"
                     << pos   << "\t" << "_" << "\t"
                     << "_"   << "\t"
                     << head  << "\t"
                     << label << "\t"
                     << "_"   << "\t" << "_" << "\n";
            }
            cout << endl;
        }
    }
    cerr << endl;
    cerr << "Proj / All = " << n_proj << " / " << trees.size() << endl;

    return 0;
}

