#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <cassert>

#include "ArcStandard.h"
#include "DependencyParser.h"
#include "Util.h"
#include "Config.h"
#include "time.h"
// #include "../utils/io.h"
// #include "../utils/logging.h"

using namespace std;

DependencyParser::DependencyParser(const char * cfg_filename)
{
    config.set_properties(cfg_filename);
}

DependencyParser::DependencyParser(string& cfg_filename)
{
    config.set_properties(cfg_filename.c_str());
}

DependencyParser::~DependencyParser()
{
    system = NULL; delete system;
    classifier = NULL; delete classifier;
}

void DependencyParser::train(
        string & train_file,
        string & dev_file,
        string & model_file,
        string & embed_file)
{
    train(train_file.c_str(),
            dev_file.c_str(),
            model_file.c_str(),
            embed_file.c_str());
}

void DependencyParser::train(
        const char * train_file,
        const char * dev_file,
        const char * model_file,
        const char * embed_file)
{
    cerr << "Train file:     " << train_file << endl;
    cerr << "Dev file:       " << dev_file   << endl;
    cerr << "Model file:     " << model_file << endl;
    cerr << "Embedding file: " << embed_file << endl;

    /**
     * collect training instances from train_file
     */
    vector<DependencyTree> train_trees;
    vector<DependencySent> train_sents;

    cerr << "Loading training file (conll)" << endl;
    Util::load_conll_file(train_file, train_sents, train_trees, config.labeled);
    Util::print_tree_stats(train_trees);

    vector<DependencyTree> dev_trees;
    vector<DependencySent> dev_sents;
    if (dev_file != NULL)
    {
        cerr << "Loading devel file (conll)" << endl;
        Util::load_conll_file(dev_file, dev_sents, dev_trees, config.labeled);
        Util::print_tree_stats(dev_trees);
    }


    // Read embedding file and initialize the embedding matrix
    cerr << "Reading word embeddings" << endl;
    read_embed_file(embed_file);


    /**
     * fill @known_words, @known_poss, @known_labels
     */
    cerr << "Generating dictionaries" << endl;
    gen_dictionaries(train_sents, train_trees);

    // TODO
    vector<string> ldict = known_labels;
    if (config.labeled)
        ldict.pop_back(); // remove the NIL label
    system = new ArcStandard(ldict, config.labeled);

    cerr << "Setup classifier for training" << endl;
    setup_classifier_for_training(train_sents, train_trees, embed_file);
    config.print_info();

    /**
     * Gradient Check
     */
    // classifier->check_gradient();
    // classifier->check_gradient();
    // return ;

    /**
     * train classifier by calling classifier.train
     */

    save_model(string(model_file)); // can rename
    double best_uas = -DBL_MAX;
    for (int iter = 0; iter < config.max_iter; ++iter)
    {
        /**
         * Compute current cost
         * and all gradients
         */
        double before = get_time();
        classifier->compute_cost_function();
        double after = get_time();
        // double cost = classifier->get_cost();
        cerr << "#Iteration " << (iter + 1) << ": "
             << "Cost = " << classifier->get_loss()
             << ", Correct(%) = " << classifier->get_accuracy()
             << " (" << (after - before) << ")"
             << endl;
        /**
         * update gradient using AdaGrad
         */
        // fix all words, except -UNKNOWN-, -NULL-, and -ROOT-
        if (config.fix_word_embeddings && !config.delexicalized)
            classifier->take_ada_gradient_step(known_words.size() - 3);
        else
            classifier->take_ada_gradient_step();

        if (dev_file != NULL &&
                iter % config.eval_per_iter == 0)
        {
            classifier->pre_compute(); // with updated weights
            vector<DependencyTree> predicted;
            predict(dev_sents, predicted);
            // double uas = system->get_uas_score(dev_sents, predicted, dev_trees);

            map<string, double> result;
            system->evaluate(dev_sents, predicted, dev_trees, result);
            double uas = result["UASwoPunc"];
            double las = result["LASwoPunc"];
            double uem = result["UEMwoPunc"];
            double root = result["ROOT"];

            cerr << "UAS(dev) = " << uas << "%" << endl;
            cerr << "LAS(dev) = " << las << "%" << endl;
            cerr << "UEM(dev) = " << uem << "%" << endl;
            cerr << "ROOT(dev) = " << root << "%" << endl;

            if (config.save_intermediate && uas > best_uas)
            {
                best_uas = uas;
                // save_model(string(model_file) + "." + to_str(iter)); // can rename
                save_model(string(model_file)); // can rename
            }
        }

        if (config.clear_gradient_per_iter > 0
                && iter % config.clear_gradient_per_iter == 0)
        {
            classifier->clear_gradient_histories();
        }
    }

    classifier->finalize_training();

    if (dev_file != NULL)
    {
        vector<DependencyTree> predicted;
        predict(dev_sents, predicted);
        // get_uas_scoredouble uas = system->get_uas_score(dev_sents, predicted, dev_trees);

        map<string, double> result;
        system->evaluate(dev_sents, predicted, dev_trees, result);
        double uas = result["UASwoPunc"];
        double las = result["LASwoPunc"];
        double uem = result["UEMwoPunc"];
        double root = result["ROOT"];

        cerr << "Final model: " << endl
             << "\tUAS = " << uas << endl
             << "\tLAS = " << las << endl
             << "\tUEM = " << uem << endl
             << "\tROOT = " << root << endl;
        cerr << "Best model: " << endl
             << "\tUAS(Best) = " << best_uas << endl;

        if (uas > best_uas)
        {
            save_model(model_file);
        }
    }
    else
    {
        save_model(model_file);
    }
}

void DependencyParser::gen_dictionaries(
        vector<DependencySent> & sents,
        vector<DependencyTree> & trees)
{
    vector<string> all_words;
    vector<string> all_poss;
    vector<string> all_labels;


    for (size_t i = 0; i < sents.size(); ++i)
    {
        /* debug
        sents[i].print_info();
        trees[i].print();
        */
        /*
        if (config.fix_word_embeddings)
        {
            for (size_t j = 0; j < sents[i].words.size(); ++j)
            {
                string word = sents[i].words[j];
                if (embed_ids.find(word) != embed_ids.end())
                    all_words.push_back(word);
            }
        }
        */
        all_words.insert(
                all_words.end(),
                sents[i].words.begin(),
                sents[i].words.end());

        all_poss.insert(
                all_poss.end(),
                sents[i].poss.begin(),
                sents[i].poss.end());
    }

    string root_label = "";
    if (config.labeled)
    {
        for (size_t i = 0; i < trees.size(); ++i)
        {
            for (int j = 1; j <= trees[i].n; ++j)
            {
                if (trees[i].get_head(j) == 0)
                    root_label = trees[i].get_label(j);
                else
                    all_labels.push_back(trees[i].get_label(j));
            }
        }
    }

    known_words  = Util::generate_dict(all_words, config.word_cut_off);
    known_poss   = Util::generate_dict(all_poss);
    known_labels = Util::generate_dict(all_labels);

    /**
     * Usage of these symbols:
     *
     *  UNKNOWN - Out-Of-Vocabulary words/poss
     *  NIL     - Non-Exist words/pos/label
     *  ROOT    - word form and pos for ROOT
     */
    known_words.push_back(Config::UNKNOWN);
    known_words.push_back(Config::NIL);
    known_words.push_back(Config::ROOT);

    known_poss.push_back(Config::UNKNOWN);
    known_poss.push_back(Config::NIL);
    known_poss.push_back(Config::ROOT);

    if (config.labeled)
    {
        known_labels.push_back(root_label);
        known_labels.push_back(Config::NIL);
    }
    else
    {
        known_labels.push_back(Config::UNKNOWN);
    }

    /**
     * find all oracle decisions, and extract dynamic features
     *  - can add other features here, aside from /distance/
     */
    cerr << "collect dynamic features (e.g. distances)" << endl;
    if (config.use_distance)
    {
        collect_dynamic_features(sents, trees);
        known_distances.push_back(Config::UNKNOWN_DIST);
    }

    generate_ids();

    cerr << config.SEPERATOR << endl;
    cerr << "#Word:  " << known_words.size()  << endl;
    cerr << "#POS:   " << known_poss.size()   << endl;

    if (config.labeled)
    {
        cerr << "#Label: " << known_labels.size() << endl;
        for (size_t i = 0; i < known_labels.size(); ++i)
            cerr << known_labels[i] << ", ";
        cerr << endl;
    }

    if (config.use_distance)
        cerr << "#Distances: " << known_distances.size() << endl;
}

void DependencyParser::collect_dynamic_features(
        vector<DependencySent> & sents,
        vector<DependencyTree> & trees)
{
    vector<int> all_distances;

    vector<string> ldict = known_labels;
    ldict.pop_back();
    ParsingSystem * monitor = new ArcStandard(ldict, config.labeled);

    for (size_t i = 0; i < sents.size(); ++i)
    {
        if (trees[i].is_projective())
        {
            Configuration c(sents[i]);
            while (!monitor->is_terminal(c))
            {
                string oracle = monitor->get_oracle(c, trees[i]);
                monitor->apply(c, oracle);
                all_distances.push_back(c.get_distance());
            }
        }
    }

    monitor = NULL; delete monitor;

    known_distances = Util::generate_dict(all_distances);
}

void DependencyParser::setup_classifier_for_training(
        vector<DependencySent> & sents,
        vector<DependencyTree> & trees,
        const char * embed_file)
{
    int E_entries = known_poss.size();
    if (config.labeled)
        E_entries += known_labels.size();
    if (!config.delexicalized)
        E_entries += known_words.size();
    if (config.use_distance)
        E_entries += known_distances.size();

    Mat<double> E(0.0,
                  E_entries,
                  config.embedding_size);
    Mat<double> W1(0.0,
                   config.hidden_size,
                   config.embedding_size * config.num_tokens);
    Vec<double> b1(0.0,
                   config.hidden_size);
    int n_actions = (config.labeled) ? (known_labels.size() * 2 - 1) : 3;
    Mat<double> W2(0.0, n_actions, config.hidden_size);

    double W1_init_range = sqrt(6.0 / (W1.nrows() + W1.ncols()));
    // Randomly initialize weight matrices / vectors
    for (int i = 0; i < W1.nrows(); ++i)
        for (int j = 0; j < W1.ncols(); ++j)
            W1[i][j] = (Util::rand_double() * 2 - 1) * W1_init_range;

    for (int i = 0; i < b1.size(); ++i)
        b1[i] = (Util::rand_double() * 2 - 1) * W1_init_range;

    double W2_init_range = sqrt(6.0 / (W2.nrows() + W2.ncols()));
    for (int i = 0; i < W2.nrows(); ++i)
        for (int j = 0; j < W2.ncols(); ++j)
            W2[i][j] = (Util::rand_double() * 2 - 1) * W2_init_range;

    // Match the embedding vocabulary with words in dictionary
    int in_embed = 0;
    for (int i = 0; i < E.nrows(); ++i)
    {
        if (config.delexicalized)
        {
            for (int j = 0; j < E.ncols(); ++j)
                E[i][j] = (Util::rand_double() * 2 - 1) * config.init_range;
            continue;
        }

        int index = -1;
        if (i < (int)known_words.size())
        {
            string word = known_words[i];
            if (embed_ids.find(word) != embed_ids.end())
                index = embed_ids[word];
            else if (embed_ids.find(str_tolower(word)) != embed_ids.end())
                index = embed_ids[str_tolower(word)];
            /**
             * if fix embeddings, then those words are all treated as UNK
             */
            /*
            else if (config.fix_embeddings
                        && word != Config::NIL
                        && word != Config::ROOT)
                index = embed_ids[Config::UNKNOWN];
            */
        }

        if (index >= 0)
        {
            ++ in_embed;
            for (int j = 0; j < E.ncols(); ++j)
                E[i][j] = embeddings[index][j];
        }
        else
        {
            for (int j = 0; j < E.ncols(); ++j)
                E[i][j] = (Util::rand_double() * 2 - 1) * config.init_range;
        }
    }

    cerr << "Found embeddings: "
         << in_embed
         << " / "
         << known_words.size()
         << endl;

    /**
     * generate training dataset and
     * determine the pre_computed ids (important)
     */
    Dataset dataset = gen_train_samples(sents, trees);

    /**
     * shuffle the dataset
     *  - little help
     */
    // cerr << "Shuffling training samples" << endl;
    // dataset.shuffle();

    /**
     * setup the classifier
     */
    cerr << "create classifier" << endl;
    classifier = new NNClassifier(config, dataset, E, W1, b1, W2, pre_computed_ids);
}

void DependencyParser::generate_ids()
{
    int index = 0;

    for (size_t i = 0; i < known_words.size(); ++i)
        word_ids[known_words[i]] = index++;

    if (config.delexicalized) // not counting words
        index = 0;

    for (size_t i = 0; i < known_poss.size();  ++i)
        pos_ids[known_poss[i]] = index++;

    // if (config.labeled)
    for (size_t i = 0; i < known_labels.size(); ++i)
        label_ids[known_labels[i]] = index++;

    if (config.use_distance)
        for (size_t i = 0; i < known_distances.size(); ++i)
            distance_ids[known_distances[i]] = index++;

    /* debug
    cerr << "word_ids" << endl;
    for (map<string, int>::iterator iter = word_ids.begin();
            iter != word_ids.end(); ++iter)
        cerr << iter->first << ": " << iter->second << ", ";
    cerr << endl;

    cerr << "pos_ids" << endl;
    for (map<string, int>::iterator iter = pos_ids.begin();
            iter != pos_ids.end(); ++iter)
        cerr << iter->first << ": " << iter->second << ", ";
    cerr << endl;

    cerr << "label_ids" << endl;
    for (map<string, int>::iterator iter = label_ids.begin();
            iter != label_ids.end(); ++iter)
        cerr << iter->first << ": " << iter->second << ", ";
    cerr << endl;
    */
}

Dataset DependencyParser::gen_train_samples(
        vector<DependencySent> & sents,
        vector<DependencyTree> & trees)
{
    Dataset ds_train(config.num_tokens, system->transitions.size());

    cerr << Config::SEPERATOR << endl;
    cerr << "Generating training examples..." << endl;
    unordered_map<int, int> tokpos_count;

    for (size_t i = 0; i < sents.size(); ++i)
    {
        // runtime info
        if (i > 0)
        {
            if (i % 1000 == 0)
                cerr << i << " ";
            if (i % 10000 == 0 || i == sents.size() - 1)
                cerr << endl;
        }

        /**
         * only use the projective trees
         *
         * TODO: non-projective trees also contain useful
         *  information for transition. try exploiting them.
         */
        if (trees[i].is_projective())
        {
            // cerr << i << " is projective" << endl;
            Configuration c(sents[i]);
            while (!system->is_terminal(c))
            {
                string oracle = system->get_oracle(c, trees[i]);

                vector<int> features = get_features(c);
                // int label = system->get_transition_id(oracle);
                vector<int> label(system->transitions.size(), -1);
                for (size_t j = 0; j < system->transitions.size(); ++j)
                {
                    string action = system->transitions[j];
                    if (action == oracle) label[j] = 1;
                    else if (system->can_apply(c, action)) label[j] = 0;
                    // else label.push_back(-1);
                }

                ds_train.add_sample(features, label);
                for (size_t j = 0; j < features.size(); ++j)
                {
                    int feature_id = features[j] * features.size() + j;
                    if (tokpos_count.find(feature_id) == tokpos_count.end())
                        tokpos_count[feature_id] = 1;
                    else
                        tokpos_count[feature_id] += 1;
                }

                system->apply(c, oracle);
            }
        }
    }

    cerr << "#Train examples: " << ds_train.n << endl;

    /**
     * Determine the pre-computed feature IDs.
     *
     * NB: both {tok, pos} are taken into acocunt
     *  in order to attain the feature ID.
     */
    // /* debug
    vector< pair<int, int> > temp(tokpos_count.begin(), tokpos_count.end());
    cerr << "sort tokpos_count" << endl;
    sort(temp.begin(), temp.end(), Util::comp_by_value_descending<int, int>);

    cerr << "fill pre_computed_ids" << endl;
    int real_size = min((int)temp.size(), config.num_pre_computed);
    for (int i = 0; i < real_size; ++i)
        pre_computed_ids.push_back(temp[i].first);
    // */

    return ds_train;
}

void DependencyParser::scan_test_samples(
        vector<DependencySent> & sents,
        vector<DependencyTree> & trees,
        vector<int> & precompute_ids)
{
    unordered_map<int, int> tokpos_count;
    precompute_ids.clear();

    for (size_t i = 0; i < sents.size(); ++i)
    {
        if (trees[i].is_projective())
        {
            Configuration c(sents[i]);
            while (!system->is_terminal(c))
            {
                string oracle = system->get_oracle(c, trees[i]);
                vector<int> features = get_features(c);

                for (size_t j = 0; j < features.size(); ++j)
                {
                    int feature_id = features[j] * features.size() + j;
                    if (tokpos_count.find(feature_id) == tokpos_count.end())
                        tokpos_count[feature_id] = 1;
                    else
                        tokpos_count[feature_id] += 1;
                }

                system->apply(c, oracle);
            }
        }
    }

    vector< pair<int, int> > temp(tokpos_count.begin(), tokpos_count.end());
    cerr << "sort tokpos_count" << endl;
    sort(temp.begin(), temp.end(), Util::comp_by_value_descending<int, int>);

    cerr << "fill pre_computed_ids" << endl;
    int real_size = min((int)temp.size(), config.num_pre_computed);
    for (int i = 0; i < real_size; ++i)
        precompute_ids.push_back(temp[i].first);
}

void DependencyParser::read_embed_file(const char * embed_file)
{
    ifstream emb_reader(embed_file);

    embeddings.resize(0, 0);
    embed_ids.clear();

    if (emb_reader.fail())
    {
        cerr << "# fail to open embedding file ("
             << embed_file << "), "
             << "randomly initialize."
             << endl;
        return ;
    }

    vector<string> lines;
    string line;
    while (getline(emb_reader, line))
    {
        // getline(emb_reader, line);
        // emb_reader >> line;
        if (line.length() != 0)
            lines.push_back(line);
    }

    int nwords = lines.size();
    int dim = split(lines[0]).size() - 1;

    cerr << "Embedding file: " << embed_file << endl
         << "#Words = " <<  nwords << endl
         << "#Dim   = " << dim
         << endl;

    if (dim != config.embedding_size)
        cerr << "ERROR: embedding dimension mismatch"
             << endl;

    embeddings.resize(nwords, dim);
    for (int i = 0; i < nwords; ++i)
        for (int j = 0; j < dim; ++j)
            embeddings[i][j] = 0;

    for (int i = 0; i < nwords; ++i)
    {
        vector<string> sep = split(lines[i]);

        embed_ids[sep[0]] = i;
        for (int j = 0; j < dim; ++j)
            embeddings[i][j] = to_double_sci(sep[j + 1]);
    }
}

void DependencyParser::save_model(const string & filename)
{
    save_model(filename.c_str());
}

void DependencyParser::save_model(const char * filename)
{
    /**
     * write model file along with pre-computed matrix
     * into the specified file.
     */
    Mat<double>& W1 = classifier->get_W1();
    Mat<double>& W2 = classifier->get_W2();
    Vec<double>& b1 = classifier->get_b1();
    Mat<double>& E  = classifier->get_E();

    ofstream output(filename);
    output << "dict=" << known_words.size() << "\n"
           << "pos=" << known_poss.size()  << "\n"
           << "label=" << known_labels.size() << "\n"
           << "dist=" << known_distances.size() << "\n"
           << "embeddingsize=" << E.ncols() << "\n"
           << "hiddensize=" << b1.size() << "\n"
           << "numtokens=" << W1.ncols() / E.ncols() << "\n"
           << "precomputed=" << pre_computed_ids.size() << "\n";

    int index = 0;
    // write word/pos/label embeddings
    if (!config.delexicalized)
        for (size_t i = 0; i < known_words.size(); ++i)
        {
            output << known_words[i];
            for (int j = 0; j < E.ncols(); ++j)
                output << " " << E[index][j];
            output << "\n";
            index += 1;
        }
    for (size_t i = 0; i < known_poss.size(); ++i)
    {
        output << known_poss[i];
        for (int j = 0; j < E.ncols(); ++j)
            output << " " << E[index][j];
        output << "\n";
        index += 1;
    }
    if (config.labeled)
        for (size_t i = 0; i < known_labels.size(); ++i)
        {
            output << known_labels[i];
            for (int j = 0; j < E.ncols(); ++j)
                output << " " << E[index][j];
            output << "\n";
            index += 1;
        }
    if (config.use_distance)
        for (size_t i = 0; i < known_distances.size(); ++i)
        {
            output << known_distances[i];
            for (int j = 0; j < E.ncols(); ++j)
                output << " " << E[index][j];
            output << "\n";
            index += 1;
        }

    // write classifier weights
    for (int j = 0; j < W1.ncols(); ++j)
    {
        output << W1[0][j];
        for (int i = 1; i < W1.nrows(); ++i)
            output << " " << W1[i][j];
        output << "\n";
    }
    output << b1[0];
    for (int i = 1; i < b1.size(); ++i)
        output << " " << b1[i];
    output << "\n";
    for (int j = 0; j < W2.ncols(); ++j)
    {
        output << W2[0][j];
        for (int i = 1; i < W2.nrows(); ++i)
            output << " " << W2[i][j];
        output << "\n";
    }

    // write pre-computed info (feature ids)
    for (size_t i = 0; i < pre_computed_ids.size(); ++i)
    {
        output << pre_computed_ids[i];
        if ((i + 1) % 100 == 0 ||
                i == pre_computed_ids.size() - 1)
            output << "\n";
        else
            output << " ";
    }

    output.close();
}

vector<int> DependencyParser::get_features(Configuration& c)
{
    vector<int> f_word;
    vector<int> f_pos;
    vector<int> f_label;

    for (int i = 2; i >= 0; --i)
    {
        int index = c.get_stack(i);
        f_word.push_back(get_word_id(c.get_word(index)));
        f_pos.push_back(get_pos_id(c.get_pos(index)));
    }

    for (int i = 0; i <= 2; ++i)
    {
        int index = c.get_buffer(i);
        f_word.push_back(get_word_id(c.get_word(index)));
        f_pos.push_back(get_pos_id(c.get_pos(index)));
    }

    for (int i = 0; i <= 1; ++i)
    {
        int k = c.get_stack(i);

        int index = c.get_left_child(k);
        f_word.push_back(get_word_id(c.get_word(index)));
        f_pos.push_back(get_pos_id(c.get_pos(index)));
        f_label.push_back(get_label_id(c.get_label(index)));

        index = c.get_right_child(k);
        f_word.push_back(get_word_id(c.get_word(index)));
        f_pos.push_back(get_pos_id(c.get_pos(index)));
        f_label.push_back(get_label_id(c.get_label(index)));

        index = c.get_left_child(k, 2);
        f_word.push_back(get_word_id(c.get_word(index)));
        f_pos.push_back(get_pos_id(c.get_pos(index)));
        f_label.push_back(get_label_id(c.get_label(index)));

        index = c.get_right_child(k, 2);
        f_word.push_back(get_word_id(c.get_word(index)));
        f_pos.push_back(get_pos_id(c.get_pos(index)));
        f_label.push_back(get_label_id(c.get_label(index)));

        index = c.get_left_child(c.get_left_child(k));
        f_word.push_back(get_word_id(c.get_word(index)));
        f_pos.push_back(get_pos_id(c.get_pos(index)));
        f_label.push_back(get_label_id(c.get_label(index)));

        index = c.get_right_child(c.get_right_child(k));
        f_word.push_back(get_word_id(c.get_word(index)));
        f_pos.push_back(get_pos_id(c.get_pos(index)));
        f_label.push_back(get_label_id(c.get_label(index)));
    }

    vector<int> features;
    if (!config.delexicalized)
        features.insert(features.end(),
                        f_word.begin(),
                        f_word.end());

    features.insert(features.end(),
                    f_pos.begin(),
                    f_pos.end());

    if (config.labeled)
        features.insert(features.end(),
                        f_label.begin(),
                        f_label.end());

    if (config.use_distance)
        features.push_back(get_distance_id(c.get_distance()));

    return features;
}

/*
Vec<int> get_features_array(Configuration& c)
{
    Vec<int> features(0, config.num_tokens);

    for (int i = 2; i >= 0; --j)
    {
        int index = c.get_stack(j);
        features[2-i] = get_word_id(c.get_word(index));
    }
}
*/

int DependencyParser::get_word_id(const string & s)
{
    // if fix_word_embeddings, then ignore cases
    string sl = s; // lower case form of s (if necessary)
    // if (config.fix_word_embeddings)
    //     sl = str_tolower(sl);

    // adapt to `delexicalized`, introduce Config::NONEXIST
    return (word_ids.find(sl) == word_ids.end())
                ? ((word_ids.find(Config::UNKNOWN) == word_ids.end())
                        ? Config::NONEXIST
                        : word_ids[Config::UNKNOWN])
                : word_ids[sl];
}

int DependencyParser::get_pos_id(const string & s)
{
    return (pos_ids.find(s) == pos_ids.end())
                ? pos_ids[Config::UNKNOWN]
                : pos_ids[s];
}

int DependencyParser::get_label_id(const string & s)
{
    return label_ids[s];
}

int DependencyParser::get_distance_id(const int d)
{
    return (distance_ids.find(d) == distance_ids.end())
                ? distance_ids[Config::UNKNOWN_DIST]
                : distance_ids[d];
}

void DependencyParser::predict(
        DependencySent& sent,
        DependencyTree& tree)
{
    int num_trans = system->transitions.size();
    Configuration c(sent);

    while (!system->is_terminal(c))
    {
        vector<double> scores;
        vector<int> features = get_features(c);
        classifier->compute_scores(features, scores);

        double opt_score = -DBL_MAX;
        string opt_trans = "";

        for (int i = 0; i < num_trans; ++i)
        {
            if (scores[i] > opt_score)
            {
                if (system->can_apply(c, system->transitions[i]))
                {
                    opt_score = scores[i];
                    opt_trans = system->transitions[i];
                    // cerr << "true: " << opt_trans << endl;
                }
                /*
                else
                {
                    cerr << system->transitions[i] << " can not apply" << endl;
                    cerr << "#configuration:" << endl
                         << "stack = " << c.stack.size() << ", " << c.get_stack(0) << " | " << c.get_stack(1) << endl
                         << "buffer = " << c.buffer.size() << ", " << c.get_buffer(0) << endl;
                }
                */
            }
        }
        // cerr << opt_trans << "->";
        system->apply(c, opt_trans);
    }

    tree = c.tree;
    // return c.tree;
}

void DependencyParser::predict(
        vector<DependencySent>& sents,
        vector<DependencyTree>& trees)
{
    // vector<DependencyTree> result;
    trees.clear();
    trees.resize(sents.size());
    for (size_t i = 0; i < sents.size(); ++i)
    {
        cerr << "\r" << i;
        predict(sents[i], trees[i]);
        // result.push_back(predict(sents[i]));
    }
    cerr << endl;
    // return result;
}

void DependencyParser::load_model(const char * filename)
{
    cerr << "Loading depparse model from " << filename << endl;

    double start = get_time();

    ifstream input(filename);
    int n_dict = 0, n_pos = 0, n_label = 0, n_dist = 0;
    int E_size = 0, h_size = 0, n_tokens = 0, n_pre_computed = 0;

    string s;
    for (int k = 0; k < 8; ++k)
    {
        getline(input, s);
        vector<string> kv = split_by_sep(s, "=");
        int val = to_int(kv[1]);
        switch (k)
        {
            case 0:
                n_dict = val;
                break;
            case 1:
                n_pos = val;
                break;
            case 2:
                n_label = val;
                break;
            case 3:
                n_dist = val;
                break;
            case 4:
                E_size = val;
                break;
            case 5:
                h_size = val;
                break;
            case 6:
                n_tokens = val;
                break;
            case 7:
                n_pre_computed = val;
                break;
            default:
                break;
        }
    }

    known_words.clear();
    known_poss.clear();
    known_labels.clear();
    known_distances.clear();

    int index = 0;
    int E_entries = n_pos + n_label;
    if (!config.delexicalized)
        E_entries += n_dict;
    if (config.use_distance)
        E_entries += n_dist;

    Mat<double> E(E_entries, E_size);

    if (!config.delexicalized)
        for (int i = 0; i < n_dict; ++i)
        {
            getline(input, s);
            vector<string> sep = split(s);
            known_words.push_back(sep[0]);
            for (int j = 0; j < E_size; ++j)
                E[index][j] = to_double_sci(sep[j+1]);
            index += 1;
        }

    for (int i = 0; i < n_pos; ++i)
    {
        getline(input, s);
        vector<string> sep = split(s);
        known_poss.push_back(sep[0]);
        for (int j = 0; j < E_size; ++j)
            E[index][j] = to_double_sci(sep[j+1]);
        index += 1;
    }
    if (config.labeled)
        for (int i = 0; i < n_label; ++i)
        {
            getline(input, s);
            vector<string> sep = split(s);
            known_labels.push_back(sep[0]);
            for (int j = 0; j < E_size; ++j)
                E[index][j] = to_double_sci(sep[j+1]);
            index += 1;
        }
    else
    {
        known_labels.push_back(Config::UNKNOWN);
    }
    if (config.use_distance)
        for (int i = 0; i < n_dist; ++i)
        {
            getline(input, s);
            vector<string> sep = split(s);
            known_distances.push_back(to_int(sep[0]));
            for (int j = 0; j < E_size; ++j)
                E[index][j] = to_double_sci(sep[j+1]);
            index += 1;
        }

    generate_ids();

    Mat<double> W1(h_size, n_tokens * E_size);
    for (int j = 0; j < W1.ncols(); ++j)
    {
        getline(input, s);
        vector<string> sep = split(s);
        for (int i = 0; i < W1.nrows(); ++i)
            W1[i][j] = to_double_sci(sep[i]);
    }

    Vec<double> b1(h_size);
    getline(input, s);
    vector<string> sep = split(s);
    for (int i = 0; i < b1.size(); ++i)
    {
        b1[i] = to_double_sci(sep[i]);
    }

    int n_actions = (config.labeled) ? (n_label * 2 - 1) : 3;
    Mat<double> W2(n_actions, h_size);
    for (int j = 0; j < W2.ncols(); ++j)
    {
        getline(input, s);
        vector<string> sep = split(s);
        for (int i = 0; i < W2.nrows(); ++i)
            W2[i][j] = to_double_sci(sep[i]);
    }

    pre_computed_ids.clear();
    while (pre_computed_ids.size() < (size_t)n_pre_computed)
    {
        getline(input, s);
        vector<string> sep = split(s);
        for (size_t i = 0; i < sep.size(); ++i)
            pre_computed_ids.push_back(to_int(sep[i]));
    }

    input.close();
    classifier = new NNClassifier(config, E, W1, b1, W2, pre_computed_ids);
    vector<string> ldict = known_labels;
    if (config.labeled)
        ldict.pop_back(); // remove the NIL label
    system = new ArcStandard(ldict, config.labeled);

    if (config.num_pre_computed > 0)
        classifier->pre_compute();

    double end = get_time();
    cerr << "Elapsed " << (end - start) << "s\n";
}

void DependencyParser::load_model(const string & filename)
{
    load_model(filename.c_str());
}

/**
 * load model trained in source language
 *  and embeddings from target language
 */
void DependencyParser::load_model_cl(
        const char * filename,
        const char * clemb)
{
    cerr << "Loading (SRC) depparse model from " << filename << endl;

    double start = get_time();

    ifstream input(filename);
    int n_dict = 0, n_pos = 0, n_label = 0, n_dist = 0;
    int E_size = 0, h_size = 0, n_tokens = 0, n_pre_computed = 0;

    string s;
    for (int k = 0; k < 8; ++k)
    {
        getline(input, s);
        vector<string> kv = split_by_sep(s, "=");
        int val = to_int(kv[1]);
        switch (k)
        {
            case 0:
                n_dict = val;
                break;
            case 1:
                n_pos = val;
                break;
            case 2:
                n_label = val;
                break;
            case 3:
                n_dist = val;
                break;
            case 4:
                E_size = val;
                break;
            case 5:
                h_size = val;
                break;
            case 6:
                n_tokens = val;
                break;
            case 7:
                n_pre_computed = val;
                break;
            default:
                break;
        }
    }

    known_words.clear();
    known_poss.clear();
    known_labels.clear();
    known_distances.clear();

    read_embed_file(clemb);
    int n_cldict = embeddings.nrows(); // vocab size of target language

    int index = 0;
    int E_entries = n_cldict + 3 + n_pos + n_label; // 3 -> -UNKNOWN-, -NULL-, -ROOT-
    if (config.use_distance)
        E_entries += n_dist;

    Mat<double> E(E_entries, E_size);
    unordered_map<string, int>::iterator iter = embed_ids.begin();
    for (; iter != embed_ids.end(); ++iter)
    {
        string word = iter->first;
        known_words.push_back(word);

        for (int j = 0; j < E_size; ++j)
            E[index][j] = embeddings[iter->second][j];
        index += 1;
    }

    cerr << "index = " << index << endl;
    cerr << "n_cldict = " << n_cldict << endl;
    assert (index == n_cldict);
    assert (n_cldict == (int)known_words.size()); // debug

    for (int i = 0; i < n_dict; ++i)
    {
        getline(input, s);
        vector<string> sep = split(s);
        if (sep[0] == Config::UNKNOWN
                || sep[0] == Config::ROOT
                || sep[0] == Config::NIL)
        {
            known_words.push_back(sep[0]);
            for (int j = 0; j < E_size; ++j)
                E[index][j] = to_double_sci(sep[j+1]);
            index += 1;
        }
        // else: skip all words in source language
    }

    assert (index == n_cldict + 3); // debug

    for (int i = 0; i < n_pos; ++i)
    {
        getline(input, s);
        vector<string> sep = split(s);
        known_poss.push_back(sep[0]);
        for (int j = 0; j < E_size; ++j)
            E[index][j] = to_double_sci(sep[j+1]);
        index += 1;
    }
    if (config.labeled)
        for (int i = 0; i < n_label; ++i)
        {
            getline(input, s);
            vector<string> sep = split(s);
            known_labels.push_back(sep[0]);
            for (int j = 0; j < E_size; ++j)
                E[index][j] = to_double_sci(sep[j+1]);
            index += 1;
        }
    else
    {
        known_labels.push_back(Config::UNKNOWN);
    }
    if (config.use_distance)
        for (int i = 0; i < n_dist; ++i)
        {
            getline(input, s);
            vector<string> sep = split(s);
            known_distances.push_back(to_int(sep[0]));
            for (int j = 0; j < E_size; ++j)
                E[index][j] = to_double_sci(sep[j+1]);
            index += 1;
        }

    generate_ids();

    Mat<double> W1(h_size, n_tokens * E_size);
    for (int j = 0; j < W1.ncols(); ++j)
    {
        getline(input, s);
        vector<string> sep = split(s);
        for (int i = 0; i < W1.nrows(); ++i)
            W1[i][j] = to_double_sci(sep[i]);
    }

    Vec<double> b1(h_size);
    getline(input, s);
    vector<string> sep = split(s);
    for (int i = 0; i < b1.size(); ++i)
    {
        b1[i] = to_double_sci(sep[i]);
    }

    int n_actions = (config.labeled) ? (n_label * 2 - 1) : 3;
    Mat<double> W2(n_actions, h_size);
    for (int j = 0; j < W2.ncols(); ++j)
    {
        getline(input, s);
        vector<string> sep = split(s);
        for (int i = 0; i < W2.nrows(); ++i)
            W2[i][j] = to_double_sci(sep[i]);
    }

    pre_computed_ids.clear();
    while (pre_computed_ids.size() < (size_t)n_pre_computed)
    {
        getline(input, s);
        vector<string> sep = split(s);
        for (size_t i = 0; i < sep.size(); ++i)
            pre_computed_ids.push_back(to_int(sep[i]));
    }

    input.close();
    classifier = new NNClassifier(config, E, W1, b1, W2, vector<int>());
    vector<string> ldict = known_labels;
    if (config.labeled)
        ldict.pop_back(); // remove the NIL label
    system = new ArcStandard(ldict, config.labeled);

    /*
    if (config.num_pre_computed > 0)
        classifier->pre_compute();
    */

    double end = get_time();
    cerr << "Elapsed " << (end - start) << "s\n";
}

void DependencyParser::load_model_cl(const string & filename, const string & clemb)
{
    load_model_cl(filename.c_str(), clemb.c_str());
}

void DependencyParser::test(
        const char * test_file,
        const char * output_file,
        bool re_precompute)
{
    // predict
    cerr << "Test file: " << test_file << endl;

    double start = get_time();

    vector<DependencySent> test_sents;
    vector<DependencyTree> test_trees;
    Util::load_conll_file(test_file, test_sents, test_trees);
    Util::print_tree_stats(test_trees);

    if (re_precompute)
    {
        cerr << "Pre-computing basedon test file (Oracle)" << endl;
        vector<int> test_precompute_ids;
        scan_test_samples(test_sents, test_trees, test_precompute_ids);
        cerr << "test_precompute_ids.size = " << test_precompute_ids.size() << endl;
        classifier->pre_compute(test_precompute_ids, true);
    }

    int n_words = 0;
    int n_sents = test_sents.size();
    for (size_t i = 0; i < test_sents.size(); ++i)
        n_words += test_sents[i].n;

    vector<DependencyTree> predicted;
    predict(test_sents, predicted);

    map<string, double> result;
    system->evaluate(test_sents, predicted, test_trees, result);
    double las_wo_punc = result["LASwoPunc"];
    double uas_wo_punc = result["UASwoPunc"];
    double uas_sub_obj = result["SOUAS"];

    fprintf(stderr, "UAS = %.4f%%\n", uas_wo_punc);
    fprintf(stderr, "LAS = %.4f%%\n", las_wo_punc);
    fprintf(stderr, "SUB/OBJ-UAS = %.4f%%\n", uas_sub_obj);

    double end = get_time();

    double wordspersec = n_words / (end - start);
    double sentspersec = n_sents / (end - start);

    fprintf(stderr, "%.1f words per second.\n", wordspersec);
    fprintf(stderr, "%.1f sents per second.\n", sentspersec);

    if (output_file != NULL)
        Util::write_conll_file(output_file, test_sents, predicted);
}

void DependencyParser::test(
        string & test_file,
        string & output_file,
        bool re_precompute)
{
    test(test_file.c_str(), output_file.c_str(), re_precompute);
}

