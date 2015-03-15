#include "ArcStandard.h"
#include "strutils.h"
#include "Config.h"

#include <vector>
using namespace std;

ArcStandard::ArcStandard(vector<string>& ldict, bool is_labeled)
{
    lang = "english"; // hardcode, TODO

    eval_corpora = "conllx"; // conllx || udt
    eval_lang = "german"; // for cross-lingual testing: german || spanish

    labeled = is_labeled;

    labels = ldict;
    root_label = labels[labels.size() - 1];
    make_transitions();

    cerr << Config::SEPERATOR << endl
         << "#Transitions: "  << transitions.size() << endl
         << "#Labels:      "  << labels.size()      << endl
         << "ROOT LABEL:   "  << root_label         << endl
         << "Labeled:      "  << labeled            << endl;
}

ArcStandard::~ArcStandard()
{
}

void ArcStandard::make_transitions()
{
    for (size_t i = 0; i < labels.size(); ++i)
        transitions.push_back("L(" + labels[i] + ")");
    for (size_t i = 0; i < labels.size(); ++i)
        transitions.push_back("R(" + labels[i] + ")");

    transitions.push_back("S"); // shift

    cerr << "Transition types:" << endl;
    for (size_t i = 0; i < transitions.size(); ++i)
        cerr << transitions[i] << ", ";
    cerr << endl;
}

bool ArcStandard::can_apply(Configuration& c, const string& t)
{
    if (startswith(t, "L") || startswith(t, "R"))
    {
        string label = t.substr(2, t.length() - 3);
        int h = startswith(t, "L") ? c.get_stack(0) : c.get_stack(1);

        if (h < 0)
            return false;
        if (labeled)
        {
            // set<string> punc_tags = get_punctuation_tags();
            // if (punc_tags.find(c.get_pos(h)) != punc_tags.end()) return false;
            if (h == 0 && !(label == root_label))   return false;

            // When training parsers for multi-rooted data, comment the following lines
            if (h > 0 && (label == root_label))     return false;
            //     cerr << "h > 0 and label == root_label" << endl;
        }
    }

    int n_stack = c.get_stack_size();
    int n_buffer = c.get_buffer_size();

    if (startswith(t, "L"))
        return n_stack > 2;
    else if (startswith(t, "R"))
        // single root
        return (n_stack > 2) || (n_stack == 2 && n_buffer == 0);
    else
        return n_buffer > 0; // shift
}

void ArcStandard::apply(Configuration& c, const string& t)
{
    int w1 = c.get_stack(1);
    int w2 = c.get_stack(0);

    // Left Reduce
    if (startswith(t, "L"))
    {
        c.add_arc(w2, w1, t.substr(2, t.length() - 3));
        c.remove_second_top_stack();
        c.lvalency[w2] += 1;
    }
    // Right Reduce
    else if (startswith(t, "R"))
    {
        c.add_arc(w1, w2, t.substr(2, t.length() - 3));
        c.remove_top_stack();
        c.rvalency[w1] += 1;
    }
    // Shift
    else
        c.shift();
}

const string ArcStandard::get_oracle(
        Configuration& c,
        DependencyTree& tree)
{
    int w1 = c.get_stack(1);
    int w2 = c.get_stack(0);

    if (w1 > 0 && tree.get_head(w1) == w2)
        return "L(" + tree.get_label(w1) + ")";
    else if (w1 >= 0
            && tree.get_head(w2) == w1
            && !c.has_other_child(w2, tree))
        return "R(" + tree.get_label(w2) + ")";
    else
        return "S";
}

bool ArcStandard::is_oracle(
        Configuration& c,
        string& t,
        DependencyTree& tree)
{
    // unused so far
    return true;
}

Configuration ArcStandard::init_configuration(DependencySent& sent)
{
    Configuration c(sent);
    return c;
}

bool ArcStandard::is_terminal(Configuration& c)
{
    return c.get_stack_size() == 1 &&
        c.get_buffer_size() == 0;
}

