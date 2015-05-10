#ifndef __NNDEP_CLASSIFIER_H__
#define __NNDEP_CLASSIFIER_H__

#include "Config.h"
#include "Dataset.h"
// #include "math/mat.h"
// #include <map>
#ifndef EIGEN_USE_MKL_ALL
    #define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unordered_map>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Cost
{
    public:
        double loss;
        double percent_correct;

        MatrixXd grad_W1;
        VectorXd grad_b1;
        MatrixXd grad_W2;
        MatrixXd grad_Eb;
        MatrixXd grad_Ed;
        MatrixXd grad_Ev;
        MatrixXd grad_Ec;

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
                MatrixXd & _grad_W1,
                VectorXd & _grad_b1,
                MatrixXd & _grad_W2,
                MatrixXd & _grad_Eb,
                MatrixXd & _grad_Ed,
                MatrixXd & _grad_Ev,
                MatrixXd & _grad_Ec,
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
        MatrixXd get_grad_W1()
        {
            return grad_W1;
        }
        VectorXd get_grad_b1()
        {
            return grad_b1;
        }
        MatrixXd get_grad_W2()
        {
            return grad_W2;
        }
        MatrixXd get_grad_Eb()
        {
            return grad_Eb;
        }
        MatrixXd get_grad_Ed()
        {
            return grad_Ed;
        }
        MatrixXd get_grad_Ev()
        {
            return grad_Ev;
        }
        MatrixXd get_grad_Ec()
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
                const MatrixXd & _Eb,
                const MatrixXd & _Ed,
                const MatrixXd & _Ev,
                const MatrixXd & _Ec,
                const MatrixXd & _W1,
                const VectorXd & _b1,
                const MatrixXd & _W2,
                const std::vector<int>& pre_computed_ids);
        NNClassifier(
                const Config& config,
                const MatrixXd & _Eb,
                const MatrixXd & _Ed,
                const MatrixXd & _Ev,
                const MatrixXd & _Ec,
                const MatrixXd & _W1,
                const VectorXd & _b1,
                const MatrixXd & _W2,
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
                MatrixXd & num_grad_W1,
                VectorXd & num_grad_b1,
                MatrixXd & num_grad_W2,
                MatrixXd & num_grad_Eb,
                MatrixXd & num_grad_Ed,
                MatrixXd & num_grad_Ev,
                MatrixXd & num_grad_Ec);
        double compute_cost();

        void take_ada_gradient_step(int Eb_start_pos = 0);

        void dropout(
                int size,
                double prob,
                std::vector<int>& active_units,
                VectorXd & mask);

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
                VectorXd & scores);

        double get_loss();
        double get_accuracy();

        MatrixXd & get_W1();
        MatrixXd & get_W2();
        VectorXd & get_b1();
        MatrixXd & get_Eb();
        MatrixXd & get_Ed();
        MatrixXd & get_Ev();
        MatrixXd & get_Ec();

        void print_info();

    private:
        /**
         * Eb: Embedding matrix for basic features
         * Ed: Embedding matrix for distance features
         * Ev: Embedding matrix for valency features
         * Ec: Embedding matrix for cluster features
         */
        static MatrixXd W1, W2, Eb, Ed, Ev, Ec;
        static VectorXd b1;

        /*
        MatrixXd grad_W1;
        VectorXd grad_b1;
        MatrixXd grad_W2;
        MatrixXd grad_E;

        double loss;
        double accuracy;
        */
        Cost cost;

        MatrixXd eg2W1, eg2W2, eg2Eb, eg2Ed, eg2Ev, eg2Ec;
        VectorXd eg2b1;

        /**
         * global grad saved
         */
        static MatrixXd grad_saved;
        static MatrixXd saved; // pre_computed;

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
