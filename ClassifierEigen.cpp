#include "ClassifierEigen.h"
#include "Util.h"

#include <chrono>
#include "ThreadPool.h"

#include "fastexp.h"

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>
#include <set>

#include <omp.h>

using namespace Eigen;
using namespace std;

// TODO Bug: fix_embedding

/**
 * Definition of static variables
 */
MatrixXd NNClassifier::grad_saved;
MatrixXd NNClassifier::saved;
unordered_map<int, int> NNClassifier::pre_map;

MatrixXd NNClassifier::W1;
VectorXd NNClassifier::b1;
MatrixXd NNClassifier::W2;

MatrixXd NNClassifier::Eb;
MatrixXd NNClassifier::Ed;
MatrixXd NNClassifier::Ev;
MatrixXd NNClassifier::Ec;

Dataset NNClassifier::dataset;

NNClassifier::NNClassifier()
{}

NNClassifier::NNClassifier(const NNClassifier & classifier)
{
    config = classifier.config;
    num_labels = classifier.num_labels;
    debug = classifier.debug;
}

NNClassifier::NNClassifier(
        const Config & _config,
        const MatrixXd & _Eb,
        const MatrixXd & _Ed,
        const MatrixXd & _Ev,
        const MatrixXd & _Ec,
        const MatrixXd & _W1,
        const VectorXd & _b1,
        const MatrixXd & _W2,
        const vector<int> & pre_computed_ids)
{
    // NNClassifier(_config, Dataset(), _E, _W1, _b1, _W2, pre_computed_ids);
    config = _config;
    Eb = _Eb;
    Ed = _Ed;
    Ev = _Ev;
    Ec = _Ec;
    W1 = _W1;
    b1 = _b1;
    W2 = _W2;

    num_labels = W2.rows();

    cursor = 0;

    // /* debug
    for (size_t i = 0; i < pre_computed_ids.size(); ++i)
    {
        pre_map[pre_computed_ids[i]] = i;
    }
    // */
}

NNClassifier::NNClassifier(
        const Config& _config,
        const Dataset& _dataset,
        const MatrixXd& _Eb,
        const MatrixXd& _Ed,
        const MatrixXd& _Ev,
        const MatrixXd& _Ec,
        const MatrixXd& _W1,
        const VectorXd& _b1,
        const MatrixXd& _W2,
        const vector<int>& pre_computed_ids)
{
    config = _config;
    dataset = _dataset;
    Eb = _Eb;
    Ed = _Ed;
    Ev = _Ev;
    Ec = _Ec;
    W1 = _W1;
    b1 = _b1;
    W2 = _W2;

    init_gradient_histories();

    num_labels = W2.rows(); // number of transitions

    cursor = 0;

    // /* debug
    for (size_t i = 0; i < pre_computed_ids.size(); ++i)
    {
        pre_map[pre_computed_ids[i]] = i;
    }
    // */

    print_info();

    /*
    grad_W1.resize(W1.rows(), W1.cols());
    grad_b1.resize(b1.size());
    grad_W2.resize(W2.rows(), W2.cols());
    grad_E.resize(E.rows(), E.cols());
    */
    grad_saved.resize(pre_map.size(), config.hidden_size);

    debug = false;
}

Cost NNClassifier::thread_proc(vector<Sample> & chunk, size_t batch_size)
{
    MatrixXd grad_W1 = MatrixXd::Zero(W1.rows(), W1.cols());
    VectorXd grad_b1 = VectorXd::Zero(b1.size());
    MatrixXd grad_W2 = MatrixXd::Zero(W2.rows(), W2.cols());
    MatrixXd grad_Eb = MatrixXd::Zero(Eb.rows(), Eb.cols());
    MatrixXd grad_Ed = MatrixXd::Zero(Ed.rows(), Ed.cols());
    MatrixXd grad_Ev = MatrixXd::Zero(Ev.rows(), Ev.cols());
    MatrixXd grad_Ec = MatrixXd::Zero(Ec.rows(), Ec.cols());

    /*
    cerr << "W1.size = " << W1.rows() << ", " << W1.cols() << endl;
    cerr << "W2.size = " << W2.rows() << ", " << W2.cols() << endl;
    cerr << "b1.size = " << b1.size() << endl;
    cerr << "E.size = " << E.rows() << ", " << E.cols() << endl;

    cerr << "W1[0][1] = " << W1[0][1] << endl;
    cerr << "W2[0][1] = " << W2[0][1] << endl;
    cerr << "b1[1] = " << b1[1] << endl;
    cerr << "E[0][1] = " << E[0][1] << endl;
    */

    double loss = 0.0;
    int correct = 0;

    vector< vector<int> > dropout_histories;

    for (size_t i = 0; i < chunk.size(); ++i)
    {
        vector<int>& features = chunk[i].get_feature();

        vector<int>& label = chunk[i].get_label();

        // feed forward the neural net
        VectorXd scores = VectorXd::Zero(num_labels);
        VectorXd hidden = VectorXd::Zero(config.hidden_size);
        VectorXd hidden3 = VectorXd::Zero(config.hidden_size);

        // Run dropout: randomly dropout some hidden units
        vector<int> active_units;
        VectorXd mask = VectorXd::Zero(config.hidden_size);
        dropout(config.hidden_size, config.dropout_prob, active_units, mask);

        if (debug)
            dropout_histories.push_back(active_units);

        // feed forward to hidden layer
        int offset = 0;
        for (int j = 0; j < config.num_tokens; ++j)
        {
            int tok = features[j]; // feature ID
            int E_index = tok;
            // feature index in @pre_map.keys()
            // considering position in input layer
            int index = tok * config.num_tokens + j;
            int feat_type = config.get_feat_type(j);

            assert (feat_type != Config::NONEXIST);
            if (feat_type == Config::DIST_FEAT)
                E_index -= Eb.rows();
            else if (feat_type == Config::VALENCY_FEAT)
                E_index -= Eb.rows() + Ed.rows();
            else if (feat_type == Config::CLUSTER_FEAT)
                E_index -= Eb.rows() + Ed.rows() + Ev.rows();

            int emb_size = config.get_embedding_size(feat_type);
            // embedding size for current token

            // /* debug
            if (pre_map.find(index) != pre_map.end())
            {
                int id = pre_map[index];
                hidden.noalias() += saved.row(id).transpose();
            }
            else
            {
            // */
                if (feat_type == Config::BASIC_FEAT)
                    hidden.noalias() += W1.block(0, offset, W1.rows(), emb_size) * Eb.row(E_index).transpose();
                else if (feat_type == Config::DIST_FEAT)
                    hidden.noalias() += W1.block(0, offset, W1.rows(), emb_size) * Ed.row(E_index).transpose();
                else if (feat_type == Config::VALENCY_FEAT)
                    hidden.noalias() += W1.block(0, offset, W1.rows(), emb_size) * Ev.row(E_index).transpose();
                else if (feat_type == Config::CLUSTER_FEAT)
                    hidden.noalias() += W1.block(0, offset, W1.rows(), emb_size) * Ec.row(E_index).transpose();
            }
            // offset += config.embedding_size;
            offset += emb_size;
        }
        // hidden.cwiseProduct(mask); // dropout

        // add bias term
        // activate
        hidden.noalias() += b1;
        // hidden.noalias() = hidden.cwiseProduct(mask); // dropout
        for (int j = 0; j < mask.size(); ++j)
            if (mask(j) == 0)
                hidden(j) = 0;

        hidden3 = hidden.cwiseProduct(hidden).cwiseProduct(hidden);

        /*
        cerr << "hidden: " << endl;
        for (int j = 0; j < hidden.size(); ++j)
            cerr << hidden[j] << " ";
        cerr << endl;

        cerr << "hidden3: " << endl;
        for (int j = 0; j < hidden.size(); ++j)
            cerr << hidden3[j] << " ";
        cerr << endl;
        */

        // feed forward to softmax layer
        scores.noalias() = W2 * hidden3;
        VectorXd::Index max_idx;
        double max_score = scores.maxCoeff(&max_idx);

        /*
        cerr << "unnormalized scores: " << endl;
        for (int j = 0; j < scores.size(); ++j)
            cerr << scores[j] << " ";
        cerr << endl;
        */

        double sum1 = .0;
        double sum2 = .0;
        for (int j = 0; j < num_labels; ++j)
        {
            if (label[j] >= 0)
            {
                scores(j) = fastexp(scores(j) - max_score);
                if (label[j] == 1) sum1 += scores(j);
                sum2 += scores(j);
            }
        }

        /*
        cerr << "normalized scores: " << endl;
        for (int j = 0; j < scores.size(); ++j)
            cerr << scores[j] << " ";
        cerr << endl;

        // cerr << "label = " << label << " | " << num_labels << endl;
        cerr << "sum1 = " << sum1 << endl;
        cerr << "sum2 = " << sum2 << endl;
        cerr << "add to cost: (" << log(sum2) << " - " << log(sum1) << ")" << endl;
        */
        loss += (log(sum2) - log(sum1)); // divide batch_size
        if (label[max_idx] == 1)
            correct += 1; // divide batch_size

        // compute the gradients
        // here, we only consider the situation where only one unit
        // in the output layer is activated.
        // NB: in Danqi's implementation, she consider all possible decisions
        VectorXd grad_hidden3 = VectorXd::Zero(config.hidden_size);
        // double delta = -(1 - scores[label] / sum2) / config.batch_size;

        for (int i = 0; i < num_labels; ++i)
        {
            if (label[i] >= 0)
            {
                double delta = -(label[i] - scores(i) / sum2) / batch_size;
                for (size_t j = 0; j < active_units.size(); ++j)
                {
                    int node_index = active_units[j];
                    grad_W2(i, node_index) += delta * hidden3(node_index);
                    grad_hidden3(node_index) += delta * W2(i, node_index);
                }
            }
        }

        VectorXd grad_hidden = VectorXd::Zero(config.hidden_size);
        // #pragma omp parallel for
        for (size_t j = 0; j < active_units.size(); ++j)
        {
            int node_index = active_units[j];
            grad_hidden(node_index) = grad_hidden3(node_index)
                                        * 3
                                        * hidden(node_index)
                                        * hidden(node_index);
            grad_b1(node_index) += grad_hidden(node_index);
        }

        offset = 0;
        for (int j = 0; j < config.num_tokens; ++j)
        {
            int tok = features[j];
            int E_index = tok;
            int index = tok * config.num_tokens + j;
            int feat_type = config.get_feat_type(j);

            assert (feat_type != Config::NONEXIST);
            if (feat_type == Config::DIST_FEAT)
                E_index -= Eb.rows();
            else if (feat_type == Config::VALENCY_FEAT)
                E_index -= Eb.rows() + Ed.rows();
            else if (feat_type == Config::CLUSTER_FEAT)
                E_index -= Eb.rows() + Ed.rows() + Ev.rows();

            int emb_size = config.get_embedding_size(feat_type);
            // /* debug
            if (pre_map.find(index) != pre_map.end())
            {
                int id = pre_map[index];
                grad_saved.row(id).noalias() += grad_hidden;
            }
            else
            {
            // */
                if (feat_type == Config::BASIC_FEAT)
                {
                    grad_W1.block(0, offset, W1.rows(), emb_size).noalias() +=
                        grad_hidden * Eb.row(E_index);
                    grad_Eb.row(E_index).noalias() +=
                        grad_hidden.transpose() *
                        W1.block(0, offset, W1.rows(), emb_size);
                }
                else if (feat_type == Config::DIST_FEAT)
                {
                    grad_W1.block(0, offset, W1.rows(), emb_size).noalias() +=
                        grad_hidden * Ed.row(E_index);
                    grad_Ed.row(E_index).noalias() +=
                        grad_hidden.transpose() *
                        W1.block(0, offset, W1.rows(), emb_size);
                }
                else if (feat_type == Config::VALENCY_FEAT)
                {
                    grad_W1.block(0, offset, W1.rows(), emb_size).noalias() +=
                        grad_hidden * Ev.row(E_index);
                    grad_Ev.row(E_index).noalias() +=
                        grad_hidden.transpose() *
                        W1.block(0, offset, W1.rows(), emb_size);
                }
                else if (feat_type == Config::CLUSTER_FEAT)
                {
                    grad_W1.block(0, offset, W1.rows(), emb_size).noalias() +=
                        grad_hidden * Ec.row(E_index);
                    grad_Ec.row(E_index).noalias() +=
                        grad_hidden.transpose() *
                        W1.block(0, offset, W1.rows(), emb_size);
                }
            }
            offset += emb_size;
        }
    }

    loss /= batch_size;
    double accuracy = (double)correct / batch_size;

    Cost cost(loss,
                accuracy,
                grad_W1,
                grad_b1,
                grad_W2,
                grad_Eb,
                grad_Ed,
                grad_Ev,
                grad_Ec,
                dropout_histories);

    /*
    cerr << "grad_W2: " << grad_W2.rows() << " * " << grad_W2.cols() << endl
         << "grad_W1: " << grad_W1.rows() << " * " << grad_W1.cols() << endl
         << "grad_Eb: " << grad_Eb.rows() << " * " << grad_Eb.cols() << endl
         << "grad_Ed: " << grad_Ed.rows() << " * " << grad_Ed.cols() << endl
         << "grad_Ev: " << grad_Ev.rows() << " * " << grad_Ev.cols() << endl
         << "grad_Ec: " << grad_Ec.rows() << " * " << grad_Ec.cols() << endl;
    */

    return cost;
}

void NNClassifier::compute_cost_function()
{
    if (debug)
        cost.dropout_histories.clear();

    /**
     * Randomly sample a subset of instances
     *  as a mini-batch.
     */
    /*
    samples = Util::get_random_subset(
            dataset.samples,
            config.batch_size);
    */
    Util::get_minibatch(
            dataset.samples,
            samples,
            config.batch_size,
            cursor);
    cursor += samples.size();
    if (cursor >= dataset.n) cursor = 0; // start over

    cerr << "Sample " << samples.size() << " samples for training" << endl;

    // should be smaller than number of CPU cores.
    int num_chunks = config.training_threads;
    vector< vector<Sample> > chunks;
    Util::partition_into_chunks(samples, chunks, num_chunks);

    /*
    for (size_t i = 0; i < chunks.size(); ++i)
        cerr << "Chunk " << i << " = " << chunks[i].size() << endl;
    */

    /**
     * determine the feature IDs which need to be pre-computed
     * for these examples
     *
     * # I think this is problematic in Danqi's code (grad_saved),
     *      since they loss the dropout information.
     *      (ok, she's right)
     */
    // /* debug
    vector<int> feature_ids_to_pre_compute = 
        get_pre_computed_ids(samples);
    pre_compute(feature_ids_to_pre_compute);
    // */

    grad_saved.setZero();

    // cerr << "build thread pool..." << endl;
    ThreadPool pool(num_chunks);
    vector< future<Cost> > results;
    for (int i = 0; i < num_chunks; ++i)
    {
        // cerr << "build " << i << "-th thread" << endl;
        results.emplace_back(
                pool.enqueue(
                    &NNClassifier::thread_proc,
                    *this,
                    chunks[i],
                    samples.size()
                )
            );
    }
    // cerr << "all threads built" << endl;

    // Merge
    cost.init();
    for (int i = 0; i < num_chunks; ++i)
    {
        if (i == 0)
            cost = results[i].get();
        else
            cost.merge(results[i].get(), debug);
    }

    // cost = 0.0;
    // int correct = 0;

    // cost /= config.batch_size;
    // cost /= samples.size();
    // accuracy = (double)correct / (double)config.batch_size;
    // accuracy = (double)correct / (double)samples.size();

    // /* debug
    back_prop_saved(cost, feature_ids_to_pre_compute);
    // */
    // cerr << "cost.grad_w1[0][0]" << cost.grad_W1[0][0] << endl;

    // cerr << "loss = " << cost.loss << endl;
    // cerr << "accuracy = " << cost.percent_correct << endl;
    add_l2_regularization(cost);
}

void NNClassifier::back_prop_saved(Cost& cost, vector<int> & features_seen)
{
    #pragma omp parallel for
    for (size_t i = 0; i < features_seen.size(); ++i)
    {
        int map_x = pre_map[features_seen[i]];
        int tok = features_seen[i] / config.num_tokens;
        // int offset = (features_seen[i] % config.num_tokens) * config.embedding_size;
        int pos = features_seen[i] % config.num_tokens;
        int feat_type = config.get_feat_type(pos);
        int offset = config.get_offset(pos);
        int emb_size = config.get_embedding_size(feat_type);

        int E_index = tok;
        assert (feat_type != Config::NONEXIST);
        if (feat_type == Config::DIST_FEAT)
            E_index -= Eb.rows();
        else if (feat_type == Config::VALENCY_FEAT)
            E_index -= Eb.rows() + Ed.rows();
        else if (feat_type == Config::CLUSTER_FEAT)
            E_index -= Eb.rows() + Ed.rows() + Ev.rows();

        VectorXd delta = grad_saved.row(map_x).transpose();

        // assert W1.rows() == config.hidden_size
        if (feat_type == Config::BASIC_FEAT)
        {
            cost.grad_W1.block(0, offset, W1.rows(), emb_size).noalias() +=
                delta * Eb.row(E_index);
            cost.grad_Eb.row(E_index).noalias() +=
                delta.transpose() * W1.block(0, offset, W1.rows(), emb_size);
        }
        else if (feat_type == Config::DIST_FEAT)
        {
            cost.grad_W1.block(0, offset, W1.rows(), emb_size).noalias() +=
                delta * Ed.row(E_index);
            cost.grad_Ed.row(E_index).noalias() +=
                delta.transpose() * W1.block(0, offset, W1.rows(), emb_size);
        }
        else if (feat_type == Config::VALENCY_FEAT)
        {
            cost.grad_W1.block(0, offset, W1.rows(), emb_size).noalias() +=
                delta * Ev.row(E_index);
            cost.grad_Ev.row(E_index).noalias() +=
                delta.transpose() * W1.block(0, offset, W1.rows(), emb_size);
        }
        else if (feat_type == Config::CLUSTER_FEAT)
        {
            cost.grad_W1.block(0, offset, W1.rows(), emb_size).noalias() +=
                delta * Ec.row(E_index);
            cost.grad_Ec.row(E_index).noalias() +=
                delta.transpose() * W1.block(0, offset, W1.rows(), emb_size);
        }
    }
}

void NNClassifier::add_l2_regularization(Cost& cost)
{
    cost.loss += config.reg_parameter * W1.squaredNorm() / 2.0;
    cost.grad_W1.noalias() += config.reg_parameter * W1;

    cost.loss += config.reg_parameter * b1.squaredNorm() / 2.0;
    cost.grad_b1.noalias() += config.reg_parameter * b1;

    cost.loss += config.reg_parameter * W2.squaredNorm() / 2.0;
    cost.grad_W2.noalias() += config.reg_parameter * W2;

    cost.loss += config.reg_parameter * Eb.squaredNorm() / 2.0;
    cost.grad_Eb.noalias() += config.reg_parameter * Eb;

    cost.loss += config.reg_parameter * Ed.squaredNorm() / 2.0;
    cost.grad_Ed.noalias() += config.reg_parameter * Ed;

    cost.loss += config.reg_parameter * Ev.squaredNorm() / 2.0;
    cost.grad_Ev.noalias() += config.reg_parameter * Ev;

    cost.loss += config.reg_parameter * Ec.squaredNorm() / 2.0;
    cost.grad_Ec.noalias() += config.reg_parameter * Ec;
}

void NNClassifier::dropout(int size, double prob, vector<int>& active_units, VectorXd & mask)
{
    active_units.clear();
    for (int i = 0; i < size; ++i)
    {
        // if (rand() % 10 / 10 > prob)
        if (Util::rand_double() > prob)
        {
            active_units.push_back(i);
            mask(i) = 1;
        }
    }
}

void NNClassifier::check_gradient()
{
    /**
     * check gradients computed by @compute_cost_function
     * with numerical gradients
     *
     * @grad_W2
     * @grad_W1
     * @grad_E
     * @grad_b1
     *
     */
    init_gradient_histories();
    cerr << "Checking Gradients..." << endl;
    // first step: randomly sample a mini-batch
    compute_cost_function(); // set cost and gradient

    MatrixXd num_grad_W1 = MatrixXd::Zero(cost.grad_W1.rows(), cost.grad_W1.cols());
    MatrixXd num_grad_W2 = MatrixXd::Zero(cost.grad_W2.rows(), cost.grad_W2.cols());
    VectorXd num_grad_b1 = VectorXd::Zero(cost.grad_b1.size());
    MatrixXd num_grad_Eb = MatrixXd::Zero(cost.grad_Eb.rows(), cost.grad_Eb.cols());
    MatrixXd num_grad_Ed = MatrixXd::Zero(cost.grad_Ed.rows(), cost.grad_Ed.cols());
    MatrixXd num_grad_Ev = MatrixXd::Zero(cost.grad_Ev.rows(), cost.grad_Ev.cols());
    MatrixXd num_grad_Ec = MatrixXd::Zero(cost.grad_Ec.rows(), cost.grad_Ec.cols());

    // second step: compute numerical gradients
    compute_numerical_gradients(
            num_grad_W1,
            num_grad_b1,
            num_grad_W2,
            num_grad_Eb,
            num_grad_Ed,
            num_grad_Ev,
            num_grad_Ec);

    // second step: compute the diff between two gradients
    // norm(numgrad-grad) / norm(numgrad+grad) should be small
    /*
    cerr << Util::l2_norm(num_grad_W1) << endl;
    cerr << Util::l2_norm(cost.grad_W1) << endl;
    double numerator = Util::l2_norm(Util::mat_subtract(num_grad_W1, cost.grad_W1));
    double denominator = Util::l2_norm(Util::mat_add(num_grad_W1, cost.grad_W1));
    cerr << numerator << "/" << denominator << endl;
    double diff_grad_W1 = numerator / denominator;
    */
    double diff_grad_W1 = (num_grad_W1 - cost.grad_W1).norm() / (num_grad_W1 + cost.grad_W1).norm();
    double diff_grad_b1 = (num_grad_b1 - cost.grad_b1).norm() / (num_grad_b1 + cost.grad_b1).norm();
    double diff_grad_W2 = (num_grad_W2 - cost.grad_W2).norm() / (num_grad_W2 + cost.grad_W2).norm();
    double diff_grad_Eb = (num_grad_Eb - cost.grad_Eb).norm() / (num_grad_Eb + cost.grad_Eb).norm();
    double diff_grad_Ed = (num_grad_Ed - cost.grad_Ed).norm() / (num_grad_Ed + cost.grad_Ed).norm();
    double diff_grad_Ev = (num_grad_Ev - cost.grad_Ev).norm() / (num_grad_Ev + cost.grad_Ev).norm();
    double diff_grad_Ec = (num_grad_Ec - cost.grad_Ec).norm() / (num_grad_Ec + cost.grad_Ec).norm();

    /*
    for (int i = 0; i < num_grad_W2.rows(); ++i)
    {
        for (int j = 0; j < num_grad_W2.cols(); ++j)
            cerr << num_grad_W2[i][j] << " ";
        cerr << endl;
    }
    */

    cerr << "diff(W1) = " << diff_grad_W1 << endl;
    cerr << "diff(b1) = " << diff_grad_b1 << endl;
    cerr << "diff(W2) = " << diff_grad_W2 << endl;
    cerr << "diff(Eb) = " << diff_grad_Eb << endl;
    cerr << "diff(Ed) = " << diff_grad_Ed << endl;
    cerr << "diff(Ev) = " << diff_grad_Ev << endl;
    cerr << "diff(Ec) = " << diff_grad_Ec << endl;
}

void NNClassifier::compute_numerical_gradients(
        MatrixXd & num_grad_W1,
        VectorXd & num_grad_b1,
        MatrixXd & num_grad_W2,
        MatrixXd & num_grad_Eb,
        MatrixXd & num_grad_Ed,
        MatrixXd & num_grad_Ev,
        MatrixXd & num_grad_Ec)
{
    if (samples.size() == 0)
    {
        cerr << "Run compute_cost_function first." << endl;
        return ;
    }

    double epsilon = 1e-4;
    cerr << "checking W1..." << endl;
    // cerr << num_grad_W1.rows() << ", " << num_grad_W1.cols() << endl;
    cerr << W1.rows() << ", " << W1.cols() << endl;
    for (int i = 0; i < W1.rows(); ++i)
        for (int j = 0; j < W1.cols(); ++j)
        {
            W1(i, j) += epsilon;
            double p_eps_cost = compute_cost();
            W1(i, j) -= 2 * epsilon;
            double n_eps_cost = compute_cost();
            num_grad_W1(i, j) = (p_eps_cost - n_eps_cost) / (2 * epsilon);
            W1(i, j) += epsilon; // reset
        }

    cerr << "checking b1..." << endl;
    for (int i = 0; i < b1.size(); ++i)
    {
        b1(i) += epsilon;
        double p_eps_cost = compute_cost();
        b1(i) -= 2 * epsilon;
        double n_eps_cost = compute_cost();
        num_grad_b1(i) = (p_eps_cost - n_eps_cost) / (2 * epsilon);
        b1(i) += epsilon;
    }

    cerr << "checking W2..." << endl;
    for (int i = 0; i < W2.rows(); ++i)
        for (int j = 0; j < W2.cols(); ++j)
        {
            W2(i, j) += epsilon;
            double p_eps_cost = compute_cost();
            W2(i, j) -= 2 * epsilon;
            double n_eps_cost = compute_cost();
            num_grad_W2(i, j) = (p_eps_cost - n_eps_cost) / (2 * epsilon);
            W2(i, j) += epsilon; // reset
        }

    cerr << "checking Eb..." << endl;
    for (int i = 0; i < Eb.rows(); ++i)
        for (int j = 0; j < Eb.cols(); ++j)
        {
            Eb(i, j) += epsilon;
            double p_eps_cost = compute_cost();
            Eb(i, j) -= 2 * epsilon;
            double n_eps_cost = compute_cost();
            num_grad_Eb(i, j) = (p_eps_cost - n_eps_cost) / (2 * epsilon);
            Eb(i, j) += epsilon; // reset
        }

    cerr << "checking Ed..." << endl;
    for (int i = 0; i < Ed.rows(); ++i)
        for (int j = 0; j < Ed.cols(); ++j)
        {
            Ed(i, j) += epsilon;
            double p_eps_cost = compute_cost();
            Ed(i, j) -= 2 * epsilon;
            double n_eps_cost = compute_cost();
            num_grad_Ed(i, j) = (p_eps_cost - n_eps_cost) / (2 * epsilon);
            Ed(i, j) += epsilon; // reset
        }

    cerr << "checking Ev..." << endl;
    for (int i = 0; i < Ev.rows(); ++i)
        for (int j = 0; j < Ev.cols(); ++j)
        {
            Ev(i, j) += epsilon;
            double p_eps_cost = compute_cost();
            Ev(i, j) -= 2 * epsilon;
            double n_eps_cost = compute_cost();
            num_grad_Ev(i, j) = (p_eps_cost - n_eps_cost) / (2 * epsilon);
            Ev(i, j) += epsilon; // reset
        }

    cerr << "checking Ec..." << endl;
    for (int i = 0; i < Ec.rows(); ++i)
        for (int j = 0; j < Ec.cols(); ++j)
        {
            Ec(i, j) += epsilon;
            double p_eps_cost = compute_cost();
            Ec(i, j) -= 2 * epsilon;
            double n_eps_cost = compute_cost();
            num_grad_Ec(i, j) = (p_eps_cost - n_eps_cost) / (2 * epsilon);
            Ec(i, j) += epsilon; // reset
        }
}

// for gradient checking. Single threading
double NNClassifier::compute_cost()
{
    // make use of samples / dropout_histories

    double v_cost = 0.0;

    // cerr << "samples.size=" << samples.size() << endl;
    // cerr << "dropout_history.size=" << cost.dropout_histories.size() << endl;

    VectorXd mask = VectorXd::Zero(config.hidden_size);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        vector<int> features = samples[i].get_feature();
        vector<int> label = samples[i].get_label();

        vector<int> active_units = cost.dropout_histories[i];
        mask.setZero();
        for (size_t j = 0; j < active_units.size(); ++j)
            mask(active_units[j]) = 1;

        VectorXd scores = VectorXd::Zero(num_labels);
        VectorXd hidden = VectorXd::Zero(config.hidden_size);
        VectorXd hidden3 = VectorXd::Zero(config.hidden_size);

        // feed-forward to hidden layer
        int offset = 0;
        for (int j = 0; j < config.num_tokens; ++j)
        {
            int tok = features[j];
            int E_index = tok;
            int feat_type = config.get_feat_type(j);

            assert (feat_type != Config::NONEXIST);
            if (feat_type == Config::DIST_FEAT)
                E_index -= Eb.rows();
            else if (feat_type == Config::VALENCY_FEAT)
                E_index -= Eb.rows() + Ed.rows();
            else if (feat_type == Config::CLUSTER_FEAT)
                E_index -= Eb.rows() + Ed.rows() + Ev.rows();

            int emb_size = config.get_embedding_size(feat_type);
            // embedding size for current token

            if (feat_type == Config::BASIC_FEAT)
                hidden += W1.block(0, offset, W1.rows(), emb_size) * Eb.row(E_index).transpose();
            else if (feat_type == Config::DIST_FEAT)
                hidden += W1.block(0, offset, W1.rows(), emb_size) * Ed.row(E_index).transpose();
            else if (feat_type == Config::VALENCY_FEAT)
                hidden += W1.block(0, offset, W1.rows(), emb_size) * Ev.row(E_index).transpose();
            else if (feat_type == Config::CLUSTER_FEAT)
                hidden += W1.block(0, offset, W1.rows(), emb_size) * Ec.row(E_index).transpose();

            offset += emb_size;
        }

        hidden += b1;
        hidden = hidden.cwiseProduct(mask);
        hidden3 = hidden.cwiseProduct(hidden).cwiseProduct(hidden);

        scores = W2 * hidden3;
        VectorXd::Index max_idx;
        double max_score = scores.maxCoeff(&max_idx);

        double sum1 = .0;
        double sum2 = .0;
        for (int j = 0; j < num_labels; ++j)
        {
            if (label[j] >= 0)
            {
                scores(j) = fastexp(scores(j) - max_score);
                if (label[j] == 1) sum1 += scores(j);
                sum2 += scores(j);
            }
        }

        v_cost += (log(sum2) - log(sum1));
    }

    v_cost /= samples.size();

    v_cost += config.reg_parameter * W1.squaredNorm() / 2.0;
    v_cost += config.reg_parameter * b1.squaredNorm() / 2.0;
    v_cost += config.reg_parameter * W2.squaredNorm() / 2.0;
    v_cost += config.reg_parameter * Eb.squaredNorm() / 2.0;
    v_cost += config.reg_parameter * Ed.squaredNorm() / 2.0;
    v_cost += config.reg_parameter * Ev.squaredNorm() / 2.0;
    v_cost += config.reg_parameter * Ec.squaredNorm() / 2.0;

    return v_cost;
}

void NNClassifier::take_ada_gradient_step(int E_start_pos)
{
    eg2W1.noalias() += cost.grad_W1.cwiseProduct(cost.grad_W1);
    W1.noalias() -= config.ada_alpha * cost.grad_W1.cwiseQuotient((eg2W1.array() + config.ada_eps).matrix().cwiseSqrt());

    eg2b1.noalias() += cost.grad_b1.cwiseProduct(cost.grad_b1);
    b1.noalias() -= config.ada_alpha * cost.grad_b1.cwiseQuotient((eg2b1.array() + config.ada_eps).matrix().cwiseSqrt());

    eg2W2.noalias() += cost.grad_W2.cwiseProduct(cost.grad_W2);
    W2.noalias() -= config.ada_alpha * cost.grad_W2.cwiseQuotient((eg2W2.array() + config.ada_eps).matrix().cwiseSqrt());

    int rcols = 0;
    if (config.fix_word_embeddings) rcols = Eb.cols() - E_start_pos;
    eg2Eb.rightCols(rcols).noalias() += cost.grad_Eb.rightCols(rcols).cwiseProduct(cost.grad_Eb.rightCols(rcols));
    Eb.rightCols(rcols).noalias() -= config.ada_alpha * cost.grad_Eb.rightCols(rcols).cwiseQuotient((eg2Eb.rightCols(rcols).array() + config.ada_eps).matrix().cwiseSqrt());

    eg2Ed.noalias() += cost.grad_Ed.cwiseProduct(cost.grad_Ed);
    Ed.noalias() -= config.ada_alpha * cost.grad_Ed.cwiseQuotient((eg2Ed.array() + config.ada_eps).matrix().cwiseSqrt());

    eg2Ev.noalias() += cost.grad_Ev.cwiseProduct(cost.grad_Ev);
    Ev.noalias() -= config.ada_alpha * cost.grad_Ev.cwiseQuotient((eg2Ev.array() + config.ada_eps).matrix().cwiseSqrt());

    eg2Ec.noalias() += cost.grad_Ec.cwiseProduct(cost.grad_Ec);
    Ec.noalias() -= config.ada_alpha * cost.grad_Ec.cwiseQuotient((eg2Ec.array() + config.ada_eps).matrix().cwiseSqrt());
}

vector<int> NNClassifier::get_pre_computed_ids(
        vector<Sample>& samples)
{
    set<int> feature_ids;

    for (size_t i = 0; i < samples.size(); ++i)
    {
        vector<int> feats = samples[i].get_feature();
        assert(feats.size() == (unsigned int)config.num_tokens);
        for (size_t j = 0; j < feats.size(); ++j)
        {
            int tok = feats[j];
            int index = tok * config.num_tokens + j;
            if (pre_map.find(index) != pre_map.end())
                feature_ids.insert(index);
        }
    }

    double percent_pre_computed =
        feature_ids.size() / (float)pre_map.size();
    cerr << "Percent necessary to pre-compute: "
         << percent_pre_computed * 100
         << "%"
         << endl;

    return vector<int>(feature_ids.begin(), feature_ids.end());
}

double NNClassifier::get_loss()
{
    return cost.loss;
}

double NNClassifier::get_accuracy()
{
    return cost.percent_correct;
}

MatrixXd & NNClassifier::get_W1()
{
    return W1;
}

MatrixXd & NNClassifier::get_W2()
{
    return W2;
}

MatrixXd & NNClassifier::get_Eb()
{
    return Eb;
}

MatrixXd & NNClassifier::get_Ed()
{
    return Ed;
}

MatrixXd & NNClassifier::get_Ev()
{
    return Ev;
}

MatrixXd & NNClassifier::get_Ec()
{
    return Ec;
}

VectorXd & NNClassifier::get_b1()
{
    return b1;
}

void NNClassifier::pre_compute()
{
    // TODO
    vector<int> candidates;
    unordered_map<int, int>::iterator iter = pre_map.begin();
    for (; iter != pre_map.end(); ++iter)
        candidates.push_back(iter->first);
    pre_compute(candidates);
}

void NNClassifier::pre_compute(
        vector<int>& candidates,
        bool refill)
{
    if (refill)
        for (size_t i = 0; i < candidates.size(); ++i)
            pre_map[candidates[i]] = i;

    // re-initialize
    saved.resize(pre_map.size(), config.hidden_size);
    saved.setZero();

    #pragma omp parallel for
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        int map_x = pre_map[candidates[i]];
        int tok = candidates[i] / config.num_tokens;
        int pos = candidates[i] % config.num_tokens;
        int feat_type = config.get_feat_type(pos);
        int offset = config.get_offset(pos);
        int emb_size = config.get_embedding_size(feat_type);

        int E_index = tok;
        assert (feat_type != Config::NONEXIST);
        if (feat_type == Config::DIST_FEAT)
            E_index -= Eb.rows();
        else if (feat_type == Config::VALENCY_FEAT)
            E_index -= Eb.rows() + Ed.rows();
        else if (feat_type == Config::CLUSTER_FEAT)
            E_index -= Eb.rows() + Ed.rows() + Ev.rows();

        if (feat_type == Config::BASIC_FEAT)
            saved.row(map_x) = Eb.row(E_index) *
                W1.block(0, offset, W1.rows(), emb_size).transpose();
        else if (feat_type == Config::DIST_FEAT)
            saved.row(map_x) = Ed.row(E_index) *
                W1.block(0, offset, W1.rows(), emb_size).transpose();
        else if (feat_type == Config::VALENCY_FEAT)
            saved.row(map_x) = Ev.row(E_index) *
                W1.block(0, offset, W1.rows(), emb_size).transpose();
        else if (feat_type == Config::CLUSTER_FEAT)
            saved.row(map_x) = Ec.row(E_index) *
                W1.block(0, offset, W1.rows(), emb_size).transpose();
    }

    cerr << "Pre-computed "
         << candidates.size()
         << endl;
}

void NNClassifier::compute_scores(
        vector<int> & features,
        VectorXd & scores)
{
    // scores.resize(num_labels);

    VectorXd hidden = VectorXd::Zero(config.hidden_size);
    int offset = 0;
    for (size_t i = 0; i < features.size(); ++i)
    {
        int tok = features[i];
        int E_index = tok;
        int index = tok * config.num_tokens + i;

        int feat_type = config.get_feat_type(i);
        int emb_size = config.get_embedding_size(feat_type);

        assert (feat_type != Config::NONEXIST);
        if (feat_type == Config::DIST_FEAT)
            E_index -= Eb.rows();
        else if (feat_type == Config::VALENCY_FEAT)
            E_index -= Eb.rows() + Ed.rows();
        else if (feat_type == Config::CLUSTER_FEAT)
            E_index -= Eb.rows() + Ed.rows() + Ev.rows();

        if (pre_map.find(index) != pre_map.end())
        {
            int id = pre_map[index];
            hidden.noalias() += saved.row(id).transpose();
        }
        else
        {
            if (feat_type == Config::BASIC_FEAT)
                hidden.noalias() += W1.block(0, offset, W1.rows(), emb_size) * Eb.row(E_index).transpose();
            else if (feat_type == Config::DIST_FEAT)
                hidden.noalias() += W1.block(0, offset, W1.rows(), emb_size) * Ed.row(E_index).transpose();
            else if (feat_type == Config::VALENCY_FEAT)
                hidden.noalias() += W1.block(0, offset, W1.rows(), emb_size) * Ev.row(E_index).transpose();
            else if (feat_type == Config::CLUSTER_FEAT)
                hidden.noalias() += W1.block(0, offset, W1.rows(), emb_size) * Ec.row(E_index).transpose();
        }
        offset += emb_size;
    }

    hidden += b1;
    hidden = hidden.cwiseProduct(hidden).cwiseProduct(hidden); // cube

    scores = W2 * hidden;
}

void NNClassifier::clear_gradient_histories()
{
    init_gradient_histories();
}

void NNClassifier::init_gradient_histories()
{
    eg2W1.resize(W1.rows(), W1.cols()); eg2W1.setZero();
    eg2W2.resize(W2.rows(), W2.cols()); eg2W2.setZero();
    eg2Eb.resize(Eb.rows(), Eb.cols()); eg2Eb.setZero();
    eg2Ed.resize(Ed.rows(), Ed.cols()); eg2Ed.setZero();
    eg2Ev.resize(Ev.rows(), Ev.cols()); eg2Ev.setZero();
    eg2Ec.resize(Ec.rows(), Ec.cols()); eg2Ec.setZero();
    eg2b1.resize(b1.size()); eg2b1.setZero();
}

void NNClassifier::finalize_training()
{
    // reset
}

void Cost::merge(const Cost & c, bool debug)
{
    loss += c.loss;
    percent_correct += c.percent_correct;
    grad_W1.noalias() += c.grad_W1;
    grad_b1.noalias() += c.grad_b1;
    grad_W2.noalias() += c.grad_W2;
    grad_Eb.noalias() += c.grad_Eb;
    grad_Ed.noalias() += c.grad_Ed;
    grad_Ev.noalias() += c.grad_Ev;
    grad_Ec.noalias() += c.grad_Ec;

    if (debug)
        dropout_histories.insert(
                dropout_histories.end(),
                c.dropout_histories.begin(),
                c.dropout_histories.end());
}

void NNClassifier::print_info()
{
    cerr << "\tW1: " << W1.rows() << " * " << W1.cols() << endl
         << "\tW2: " << W2.rows() << " * " << W2.cols() << endl
         << "\tb1: " << b1.size() << endl
         << "\tEb: " << Eb.rows() << " * " << Eb.cols() << endl
         << "\tEd: " << Ed.rows() << " * " << Ed.cols() << endl
         << "\tEv: " << Ev.rows() << " * " << Ev.cols() << endl
         << "\tEc: " << Ec.rows() << " * " << Ec.cols() << endl;
}

