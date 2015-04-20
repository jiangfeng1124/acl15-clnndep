#ifndef __NNDEP_CONFIGURATION_H__
#define __NNDEP_CONFIGURATION_H__

#include <vector>

#include "DependencyTree.h"
#include "DependencySent.h"

class Configuration
{
    public:
        Configuration() {}
        Configuration(Configuration& c);
        Configuration(DependencySent& s);
        ~Configuration() {}

        void init(DependencySent& s);

        /**
         * Shift-Reduce Actions
         */
        // shift element from buffer to queue
        bool shift();

        // remove top elements in stack
        bool remove_top_stack();

        // remove second top elements in stack
        bool remove_second_top_stack();

        int get_stack_size();

        int get_buffer_size();

        int get_sent_size();

        int get_head(int k);

        const std::string & get_label(int k);

        int get_stack(int k);

        int get_buffer(int k);

        int get_distance();

        std::string get_lvalency(int k);
        std::string get_lvalency_fc(int k); // for debug

        std::string get_rvalency(int k);
        std::string get_rvalency_fc(int k); // for debug

        std::string get_word(int k);

        std::string get_pos(int k);

        std::string get_cluster(int k);

        std::string get_cluster_prefix(int k, int p);

        void add_arc(int h, int m, const std::string & l);

        int get_left_child(int k, int cnt);

        int get_left_child(int k);

        int get_right_child(int k, int cnt);

        int get_right_child(int k);

        bool has_other_child(int k, DependencyTree& gold_tree);

        int get_left_valency(int k);

        int get_right_valency(int k);

        DependencyTree get_tree();

        std::string info();

    private:
        int encode_distance(const int & h, const int & m);
        std::string encode_valency(const std::string & typ, const int & k);

    public:
        /**
         * Not sure which one of [vector/list]
         *  is more efficient. try vector first.
         */
        std::vector<int> stack;
        std::vector<int> buffer;

        DependencyTree tree;
        DependencySent sent;

        std::vector<int> lvalency;
        std::vector<int> rvalency;
};

#endif
