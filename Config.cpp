#include "Config.h"
#include "strutils.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <map>

using namespace std;

const string Config::UNKNOWN   = "-UNKNOWN-";
const string Config::ROOT      = "-ROOT-";
const string Config::NIL       = "-NULL-";
const string Config::SEPERATOR = "#######";
const int Config::NONEXIST     = -1;
const int Config::UNKNOWN_INT  = -1;

Config::Config()
{
    init();
}

Config::Config(const char * filename)
{
    init();
    set_properties(filename);
}

void Config::init()
{
    training_threads        = 1;
    word_cut_off            = 1;
    init_range              = 0.010;
    max_iter                = 20000;
    batch_size              = 10000;
    ada_eps                 = 1.0e-6;
    ada_alpha               = 0.010;
    reg_parameter           = 1.0e-8;
    dropout_prob            = 0.50;
    hidden_size             = 200;
    embedding_size          = 50;
    num_tokens              = 48;
    num_pre_computed        = 100000;
    eval_per_iter           = 100;
    clear_gradient_per_iter = 0;
    save_intermediate       = true;
    fix_word_embeddings     = false;
    delexicalized           = false;
    labeled                 = true;
    use_distance            = false;
    distance_embedding_size = 10;
    use_valency             = false;
    valency_embedding_size  = 10;
    use_cluster             = false;
    cluster_embedding_size  = 50;
}

void Config::set_properties(const char * filename)
{
    ifstream cfg_reader(filename);
    if (cfg_reader.fail())
    {
        cerr << "# fail to open config file,"
             << " use default configuration."
             << endl;
        return ;
    }

    string line;
    map<string, string> props;
    while (getline(cfg_reader, line))
    {
        // getline(cfg_reader, line);
        // cfg_reader >> line;
        line = cutoff(line, "#");
        if (line.size() == 0)
            continue;

        vector<string> sep = split_by_sep(line, "=");
        if (sep.size() != 2)
            continue;

        sep[0] = chomp(sep[0]);
        sep[1] = chomp(sep[1]);

        props[sep[0]] = sep[1];
    }

    cfg_set_int(props, "training_threads", training_threads);
    cfg_set_int(props, "word_cut_off",     word_cut_off);
    cfg_set_int(props, "max_iter",         max_iter);
    cfg_set_int(props, "batch_size",       batch_size);
    cfg_set_int(props, "hidden_size",      hidden_size);
    cfg_set_int(props, "embedding_size",   embedding_size);
    cfg_set_int(props, "num_tokens",       num_tokens);
    cfg_set_int(props, "num_pre_computed", num_pre_computed);
    cfg_set_int(props, "eval_per_iter",    eval_per_iter);
    cfg_set_int(props, "clear_gradient_per_iter", clear_gradient_per_iter);
    cfg_set_int(props, "distance_embedding_size", distance_embedding_size);
    cfg_set_int(props, "valency_embedding_size",  valency_embedding_size);
    cfg_set_int(props, "cluster_embedding_size",  cluster_embedding_size);

    cfg_set_double(props, "init_range",     init_range);
    cfg_set_double(props, "ada_eps",        ada_eps);
    cfg_set_double(props, "ada_alpha",      ada_alpha);
    cfg_set_double(props, "reg_parameter",  reg_parameter);
    cfg_set_double(props, "dropout_prob",   dropout_prob);

    cfg_set_boolean(props, "save_intermediate", save_intermediate);
    cfg_set_boolean(props, "fix_word_embeddings", fix_word_embeddings);
    cfg_set_boolean(props, "delexicalized", delexicalized);
    cfg_set_boolean(props, "labeled", labeled);
    cfg_set_boolean(props, "use_distance", use_distance);
    cfg_set_boolean(props, "use_valency", use_valency);
    cfg_set_boolean(props, "use_cluster", use_cluster);
}

void Config::cfg_set_int(
        map<string, string>& props,
        const char * name,
        int& variable)
{
    if (props.find(name) == props.end()) // nor found
        return ;
    string value = props[name];
    if (!is_int(value))
        return ;
    variable = to_int(value);
}

void Config::cfg_set_double(
        map<string, string>& props,
        const char * name,
        double& variable)
{
    if (props.find(name) == props.end()) // nor found
        return ;
    string value = props[name];
    if (!is_double_sci(value))
        return ;
    variable = to_double_sci(props[name]);
}

void Config::cfg_set_boolean(
        map<string, string>& props,
        const char * name,
        bool& variable)
{
    if (props.find(name) == props.end()) // nor found
        return ;
    string value = props[name];
    if (!is_boolean(value))
        return ;
    variable = to_boolean(props[name]);
}

Config::Config(const string& filename)
{
    Config(filename.c_str());
}

void Config::print_info()
{
    cerr << "training_threads        = " << training_threads        << endl;
    cerr << "word_cut_off            = " << word_cut_off            << endl;
    cerr << "init_range              = " << init_range              << endl;
    cerr << "max_iter                = " << max_iter                << endl;
    cerr << "batch_size              = " << batch_size              << endl;
    cerr << "ada_eps                 = " << ada_eps                 << endl;
    cerr << "ada_alpha               = " << ada_alpha               << endl;
    cerr << "reg_parameter           = " << reg_parameter           << endl;
    cerr << "dropout_prob            = " << dropout_prob            << endl;
    cerr << "hidden_size             = " << hidden_size             << endl;
    cerr << "embedding_size          = " << embedding_size          << endl;
    cerr << "num_tokens              = " << num_tokens              << endl;
    cerr << "num_pre_computed        = " << num_pre_computed        << endl;
    cerr << "eval_per_iter           = " << eval_per_iter           << endl;
    cerr << "save_intermediate       = " << save_intermediate       << endl;
    cerr << "clear_gradient_per_iter = " << clear_gradient_per_iter << endl;
    cerr << "fix_word_embeddings     = " << fix_word_embeddings     << endl;
    cerr << "delexicalized           = " << delexicalized           << endl;
    cerr << "labeled                 = " << labeled                 << endl;
    cerr << "use_distance            = " << use_distance            << endl;
    cerr << "use_valency             = " << use_valency             << endl;
    cerr << "use_cluster             = " << use_cluster             << endl;
    cerr << "distance_embedding_size = " << distance_embedding_size << endl;
    cerr << "valency_embedding_size  = " << valency_embedding_size  << endl;
    cerr << "cluster_embedding_size  = " << cluster_embedding_size  << endl;
}

int test_config(int argc, char** argv)
{
    Config cfg("nndep.cfg");
    cfg.print_info();

    return 0;
}
