#include "Configuration.h"
#include "Config.h"
#include "strutils.h"
#include <cassert>

using namespace std;

Configuration::Configuration(Configuration& c)
{
    stack  = c.stack;
    buffer = c.buffer;
    tree   = c.tree;
    sent   = c.sent;

    lvalency = c.lvalency;
    rvalency = c.rvalency;
}

Configuration::Configuration(DependencySent& s)
{
    init(s);
}

void Configuration::init(DependencySent& s)
{
    stack.clear();
    buffer.clear();

    lvalency.clear();
    rvalency.clear();

    sent = s;
    for (int i = 1; i <= sent.n; ++i)
    {
        tree.add(Config::NONEXIST, Config::UNKNOWN);
        buffer.push_back(i);
    }

    lvalency.resize(sent.n + 1, 0);
    rvalency.resize(sent.n + 1, 0);

    stack.push_back(0);
}

bool Configuration::shift()
{
    int k = get_buffer(0);
    if (k == Config::NONEXIST)
        return false;

    buffer.erase(buffer.begin());
    stack.push_back(k);

    return true;
}

bool Configuration::remove_top_stack()
{
    int n_stack = get_stack_size();
    if (n_stack < 1)
        return false;
    stack.erase(stack.begin() + stack.size() - 1);
    return true;
}

bool Configuration::remove_second_top_stack()
{
    int n_stack = get_stack_size();
    if (n_stack < 2)
        return false;
    stack.erase(stack.begin() + stack.size() - 2);
    return true;
}

int Configuration::get_stack_size()
{
    return stack.size();
}

int Configuration::get_buffer_size()
{
    return buffer.size();
}

int Configuration::get_sent_size()
{
    return sent.n;
}

int Configuration::get_head(int k)
{
    return tree.get_head(k);
}

const string & Configuration::get_label(int k)
{
    return tree.get_label(k);
}

/**
 * k starts from 0 (top-stack)
 */
int Configuration::get_stack(int k)
{
    int n_stack = get_stack_size();
    return (k >= 0 && k < n_stack)
                ? stack[n_stack - 1 - k]
                : Config::NONEXIST;
}

int Configuration::get_buffer(int k)
{
    int n_buffer = get_buffer_size();
    return (k >= 0 && k < n_buffer)
                ? buffer[k]
                : Config::NONEXIST;
}

int Configuration::get_distance()
{
    // return abs(get_stack(0) - get_buffer(0));
    return encode_distance(get_stack(0), get_stack(1));
}

int Configuration::encode_distance(const int & h, const int & m)
{
    int diff;
    diff = h - m;
    assert(diff != 0);

    if (diff < 0) diff = -diff;
    if (diff > 10) diff = 6;
    else if (diff > 5) diff = 5;

    return diff;
}

string Configuration::encode_valency(const string & typ, const int & k)
{
    int v = k;
    if (v > 10) v = 6;
    else if (v > 5) v = 5;

    return typ + to_str(v);
}

/**
 * k starts from 0 (root)
 */
string Configuration::get_word(int k)
{
    if (k == 0)
        return Config::ROOT;
    else
        -- k;

    return (k < 0 || k >= sent.n)
                ? Config::NIL
                : sent.words[k];
}

/**
 * k starts from 0 (root)
 */
string Configuration::get_pos(int k)
{
    if (k == 0)
        return Config::ROOT;
    else
        -- k;

    return (k < 0 || k >= sent.n)
                ? Config::NIL
                : sent.poss[k];
}

string Configuration::get_cluster(int k)
{
    if (k == 0)
        return Config::ROOT;
    else
        -- k;

    return (k < 0 || k >= sent.n)
                ? Config::NIL
                : sent.clusters[k];
}

string Configuration::get_cluster_prefix(int k, int p)
{
    if (k == 0)
        return Config::ROOT;
    else
        -- k;

    return (k < 0 || k > sent.n)
                ? Config::NIL
                : get_brown_prefix(sent.clusters[k], p);
                // : ((sent.clusters[k] == Config::UNKNOWN)
                //         ? Config::UNKNOWN
                //         : get_brown_prefix(sent.clusters[k], p));
}

void Configuration::add_arc(int h, int m, const string & l)
{
    tree.set(m, h, l);
}

string Configuration::get_lvalency(int k)
{
    if (k < 0 || k > tree.n)
        return Config::UNKNOWN;

    return encode_valency("L", lvalency[k]);
}

string Configuration::get_lvalency_fc(int k)
{
    if (k < 0 || k > tree.n)
        return Config::UNKNOWN;
        // return Config::NONEXIST;

    int cnt = 0;
    for (int i = 1; i < k; ++i)
        if (tree.get_head(i) == k)
            cnt += 1;
    return "L" + to_str(cnt);
}

string Configuration::get_rvalency(int k)
{
    if (k < 0 || k > tree.n)
        return Config::UNKNOWN;

    return encode_valency("R", rvalency[k]);
}

string Configuration::get_rvalency_fc(int k)
{
    if (k < 0 || k > tree.n)
        return Config::UNKNOWN;
        // return Config::NONEXIST;

    int cnt = 0;
    for (int i = tree.n; i > k; --i)
        if (tree.get_head(i) == k)
            cnt += 1;
    return "R" + to_str(cnt);
}

int Configuration::get_left_child(int k, int cnt)
{
    if (k < 0 || k > tree.n)
        return Config::NONEXIST;

    int c = 0;
    for (int i = 1; i < k; ++i)
        if (tree.get_head(i) == k)
            if ((++c) == cnt)
                return i;
    return Config::NONEXIST;
}

int Configuration::get_left_child(int k)
{
    return get_left_child(k, 1);
}

int Configuration::get_right_child(int k, int cnt)
{
    if (k < 0 || k > tree.n)
        return Config::NONEXIST;

    int c = 0;
    for (int i = tree.n; i > k; --i)
        if (tree.get_head(i) == k)
            if ((++c) == cnt)
                return i;
    return Config::NONEXIST;
}

int Configuration::get_right_child(int k)
{
    return get_right_child(k, 1);
}

bool Configuration::has_other_child(int k, DependencyTree& gold_tree)
{
    for (int i = 1; i <= tree.n; ++i)
        if (gold_tree.get_head(i) == k
                && tree.get_head(i) != k)
            return true;
    return false;
}

int Configuration::get_left_valency(int k)
{
    if (k < 0 || k >= tree.n)
        return Config::NONEXIST;
    int cnt = 0;
    for (int i = 0; i < k; ++k)
        if (tree.get_head(i) == k)
            ++ cnt;
    return cnt;
}

int Configuration::get_right_valency(int k)
{
    if (k < 0 || k >= tree.n)
        return Config::NONEXIST;
    int cnt = 0;
    for (int i = k + 1; i <= tree.n; ++k)
        if (tree.get_head(i) == k)
            ++ cnt;
    return cnt;
}

DependencyTree Configuration::get_tree()
{
    return tree;
}

string Configuration::info()
{
    string s = "[S]";
    for (int i = 0; i < get_stack_size(); ++i)
    {
        if (i > 0)
            s.append(",");
        s += stack[i];
    }

    s.append("\n[B]");
    for (int i = 0; i < get_buffer_size(); ++i)
    {
        if (i > 0)
            s.append(",");
        s += buffer[i];
    }

    s.append("\n[H]");
    for (int i = 1; i <= tree.n; ++i)
    {
        if (i > 1)
            s.append(",");
        s.append(to_str(get_head(i)))
         .append("(")
         .append(get_label(i))
         .append(")");
    }

    return s;
}

