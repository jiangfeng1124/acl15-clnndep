/**
 *
 * Neural network based transition-based dependency parser
 * (v 0.1)
 * Author: Jiang Guo (jguo29@jhu.edu)
 * Date: 12.01.2014
 *
 */

#include "DependencyParser.h"
#include <cstring>
#include <cstdlib>
#include <ctime>

using namespace std;

#define MAX_STRING 100;

typedef struct
{
    // training mode
    bool   is_train;
    // testing mode
    bool   is_test;
    // cross-lingual testing mode
    bool   is_cltest;

    string train_file;
    string dev_file;
    string test_file;
    string model_file;
    string emb_file;
    string clemb_file;
    string cfg_file;
    string output_file;

} Option;

Option opt;

/**
 * Expected usage:
 *
 * ./nndep [options]
 *   -train  <train-file>
 *   -test   <test-file>
 *   -dev    <dev-file>
 *   -model  <model-file>
 *   -emb    <embedding-file>
 *   -cfg    <cfg-file>
 *   -output <output-file>
 */

void print_usage()
{
    cerr << "Neural Network based Transition-based Dependency Parser"
         << "(v 0.1)\n\n"
         << "Options:\n"
         << "\t-train <file>\n"
         << "\t\tUse <file> for training the model (CoNLL format)\n"
         << "\t-test <file>\n"
         << "\t\tUse <file> for testing the model (CoNLL format)\n"
         << "\t-dev <file>\n"
         << "\t\tUse <file> for evaluation during training (CoNLL format)\n"
         << "\t-model <file>\n"
         << "\t\tUse <file> for saving model\n"
         << "\t-emb <file>\n"
         << "\t\tPre-trained word embeddings, for initialization\n"
         << "\t-cltest <file>\n"
         << "\t\tUse <file> for cross-lingual testing (CoNLL format)\n"
         << "\t-clemb <file>\n"
         << "\t\tPre-trained target-language word embeddings\n"
         << "\t-cfg <file>\n"
         << "\t\tConfig file for training neural network classifier\n"
         << "\t-output <file>\n"
         << "\t\tOutput predicted parsing trees to <file>\n"
         << "\nExample(train):\n"
         << "./nndep -train data/train.dep -dev data/dev.dep"
         <<        " -model model -emb data/words.emb -cfg nndep.cfg\n"
         << "\nExample(test):\n"
         << "./nndep -test data/test.dep -model model\n\n";
}

int arg_pos(char * str, int argc, char ** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        if (!strcmp(str, argv[i]))
        {
            if (i == argc - 1)
            {
                cerr << "Argument missing for "
                     << str
                     << endl;
                exit(1);
            }
            return i;
        }
    }

    return -1;
}

void parse_command_line(int argc, char ** argv)
{
    opt.is_train = false;
    opt.is_test  = false;
    opt.is_cltest = false;
    opt.model_file = "model";

    int i;
    if ((i = arg_pos((char *)"-train",  argc, argv)) > 0)
    {
        opt.is_train = true;
        opt.train_file = argv[i + 1];
    }
    if ((i = arg_pos((char *)"-dev",    argc, argv)) > 0)
        opt.dev_file = argv[i + 1];
    if ((i = arg_pos((char *)"-test",   argc, argv)) > 0)
    {
        opt.is_test = true;
        opt.test_file = argv[i + 1];
    }
    if ((i = arg_pos((char *)"-cltest", argc, argv)) > 0)
    {
        opt.is_cltest = true;
        opt.test_file = argv[i + 1];
    }
    if ((i = arg_pos((char *)"-clemb", argc, argv)) > 0)
        opt.clemb_file = argv[i + 1];
    if ((i = arg_pos((char *)"-model",  argc, argv)) > 0)
        opt.model_file = argv[i + 1];
    if ((i = arg_pos((char *)"-emb",    argc, argv)) > 0)
        opt.emb_file = argv[i + 1];
    if ((i = arg_pos((char *)"-cfg",    argc, argv)) > 0)
        opt.cfg_file = argv[i + 1];
    if ((i = arg_pos((char *)"-output", argc, argv)) > 0)
        opt.output_file = argv[i + 1];
}

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        print_usage();
        exit(1);
    }

    parse_command_line(argc, argv);
    cerr << opt.cfg_file << endl;

    srand(time(NULL));

    DependencyParser parser(opt.cfg_file);

    bool loaded = false;
    if (opt.is_train)
    {
        parser.train(opt.train_file,
                opt.dev_file,
                opt.model_file,
                opt.emb_file);
        loaded = true;
    }

    if (opt.is_test)
    {
        if (! loaded)
            parser.load_model(opt.model_file);
        parser.test(opt.test_file,
                opt.output_file);
        // parser.save_model("tmp");
    }

    if (opt.is_cltest)
    {
        parser.load_model_cl(opt.model_file, opt.clemb_file);
        parser.test(opt.test_file,
                opt.output_file,
                true);
        // parser.save_model("tmp");
    }

    return 0;
}

