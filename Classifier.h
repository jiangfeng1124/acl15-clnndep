#ifndef __NNDEP_CLASSIFIER_H__
#define __NNDEP_CLASSIFIER_H__

#include "Config.h"
#include "Dataset.h"
#include "math/mat.h"
// #include <map>
#include <unordered_map>

class Cost
{
    public:
        double loss;
        double percent_correct;

        Mat<double> grad_W1;
        Vec<double> grad_b1;
        Mat<double> grad_W2;
        Mat<double> grad_Eb;
        Mat<double> grad_Ed;
        Mat<double> grad_Ev;
        Mat<double> grad_Ec;

        std::vector< std::vector<int> > dropout_histories;

    public:
        Cost() { init(); }

        void init()
        {
            loss = 0.0;
            percent_correct = 0.0;
        }

        Cost(const Cost & c)
        {
            loss = c.loss;
            percent_correct = c.percent_correct;

            grad_W1 = c.grad_W1;
            grad_b1 = c.grad_b1;
            grad_W2 = c.grad_W2;
            grad_Eb = c.grad_Eb;
            grad_Ed = c.grad_Ed;
            grad_Ev = c.grad_Ev;
            grad_Ec = c.grad_Ec;

            dropout_histories = c.dropout_histories;
        }

        Cost & operator= (const Cost & c)
        {
            loss = c.loss;
            percent_correct = c.percent_correct;

            grad_W1 = c.grad_W1;
            grad_b1 = c.grad_b1;
            grad_W2 = c.grad_W2;
            grad_Eb = c.grad_Eb;
            grad_Ed = c.grad_Ed;
            grad_Ev = c.grad_Ev;
            grad_Ec = c.grad_Ec;

            dropout_histories = c.dropout_histories;

            return *this;
        }

        Cost(double _loss,
                double _percent_correct,
                Mat<double>& _grad_W1,
                Vec<double>& _grad_b1,
                Mat<double>& _grad_W2,
                Mat<double>& _grad_Eb,
                Mat<double>& _grad_Ed,
                Mat<double>& _grad_Ev,
                Mat<double>& _grad_Ec,
                std::vector< std::vector<int> >& _dropout_histories)
        {
            loss = _loss;
            percent_correct = _percent_correct;
            grad_W1 = _grad_W1;
            grad_b1 = _grad_b1;
            grad_W2 = _grad_W2;
            grad_Eb = _grad_Eb;
            grad_Ed = _grad_Ed;
            grad_Ev = _grad_Ev;
            grad_Ec = _grad_Ec;
            dropout_histories = _dropout_histories;
        }

        void merge(const Cost & c, bool debug = false);

        double get_loss()
        {
            return loss;
        }
        double get_percent_correct()
        {
            return percent_correct;
        }
        Mat<double> get_grad_W1()
        {
            return grad_W1;
        }
        Vec<double> get_grad_b1()
        {
            return grad_b1;
        }
        Mat<double> get_grad_W2()
        {
            return grad_W2;
        }
        Mat<double> get_grad_Eb()
        {
            return grad_Eb;
        }
        Mat<double> get_grad_Ed()
        {
            return grad_Ed;
        }
        Mat<double> get_grad_Ev()
        {
            return grad_Ev;
        }
        Mat<double> get_grad_Ec()
        {
            return grad_Ec;
        }
};

class NNClassifier
{
    public:
        NNClassifier();
        NNClassifier(
                const Config& config,
                const Dataset& dataset,
                const Mat<double>& _Eb,
                const Mat<double>& _Ed,
                const Mat<double>& _Ev,
                const Mat<double>& _Ec,
                const Mat<double>& _W1,
                const Vec<double>& _b1,
                const Mat<double>& _W2,
                const std::vector<int>& pre_computed_ids);
        NNClassifier(
                const Config& config,
                const Mat<double>& _Eb,
                const Mat<double>& _Ed,
                const Mat<double>& _Ev,
                const Mat<double>& _Ec,
                const Mat<double>& _W1,
                const Vec<double>& _b1,
                const Mat<double>& _W2,
                const std::vector<int>& pre_computed_ids);
        NNClassifier(const NNClassifier & classifier);

        ~NNClassifier() {}// TODO

        void init_gradient_histories();

        void compute_cost_function();

        Cost thread_proc(
                std::vector<Sample> & chunk,
                size_t batch_size);

        /**
         * Gradient Checking
         */
        void check_gradient();
        void compute_numerical_gradients(
                Mat<double> & num_grad_W1,
                Vec<double> & num_grad_b1,
                Mat<double> & num_grad_W2,
                Mat<double> & num_grad_Eb,
                Mat<double> & num_grad_Ed,
                Mat<double> & num_grad_Ev,
                Mat<double> & num_grad_Ec);
        double compute_cost();

        void take_ada_gradient_step(int Eb_start_pos = 0);

        void dropout(
                int size,
                double prob,
                std::vector<int>& active_units);

        void back_prop_saved(
                Cost & cost,
                std::vector<int> & features_seen);

        void add_l2_regularization(Cost & cost);

        void clear_gradient_histories();

        void finalize_training();

        std::vector<int> get_pre_computed_ids(
                std::vector<Sample>& samples);

        void pre_compute();
        /**
         * if refill is true, then reset pre_map
         *  when testing with standard trees with answers
         *
         * Should be used after scan_test_samples(...)
         *  which collects candidate features in test data
         */
        void pre_compute(
                std::vector<int>& candidates,
                bool refill = false);

        void compute_scores(std::vector<int>& features,
                std::vector<double>& scores);

        double get_loss();
        double get_accuracy();

        Mat<double>& get_W1();
        Mat<double>& get_W2();
        Vec<double>& get_b1();
        Mat<double>& get_Eb();
        Mat<double>& get_Ed();
        Mat<double>& get_Ev();
        Mat<double>& get_Ec();

    private:
        /**
         * Eb: Embedding matrix for basic features
         * Ed: Embedding matrix for distance features
         * Ev: Embedding matrix for valency features
         * Ec: Embedding matrix for cluster features
         */
        static Mat<double> W1, W2, Eb, Ed, Ev, Ec;
        static Vec<double> b1;

        /*
        Mat<double> grad_W1;
        Vec<double> grad_b1;
        Mat<double> grad_W2;
        Mat<double> grad_E;

        double loss;
        double accuracy;
        */
        Cost cost;

        Mat<double> eg2W1, eg2W2, eg2Eb, eg2Ed, eg2Ev, eg2Ec;
        Vec<double> eg2b1;

        /**
         * global grad saved
         */
        static Mat<double> grad_saved;
        static Mat<double> saved; // pre_computed;

        /**
         * map feature ID to index in pre_computed data
         */
        static std::unordered_map<int, int> pre_map;

        bool is_training;
        int num_labels; // number of transitions

        Config config;
        static Dataset dataset; // entire dataset

        std::vector<Sample> samples; // a mini-batch
        // std::vector< std::vector<int> > dropout_histories;

        bool debug; // for gradient_checking

        int cursor; // for sampling minibatch
};


#endif
