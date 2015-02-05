#ifndef __NNDEP_PARSING_SYSTEM_H__
#define __NNDEP_PARSING_SYSTEM_H__

#include <vector>
#include <map>
#include <set>

#include "Configuration.h"
#include "DependencySent.h"
#include "DependencyTree.h"
#include "Config.h"

class ParsingSystem
{
    public:
        // ParsingSystem(std::vector<std::string>& ldict);

        int get_transition_id(const std::string & s);

        void evaluate(
                std::vector<DependencySent>& sents,
                std::vector<DependencyTree>& pred_trees,
                std::vector<DependencyTree>& gold_trees,
                std::map<std::string, double>& result);

        double get_uas_score(
                std::vector<DependencySent>& sents,
                std::vector<DependencyTree>& pred_trees,
                std::vector<DependencyTree>& gold_trees);

        std::set<std::string> get_punctuation_tags();
        std::set<std::string> get_conll_sub_obj_relations();
        std::set<std::string> get_udt_sub_obj_relations();

        virtual ~ParsingSystem();

        virtual void make_transitions() = 0;

        virtual bool can_apply(Configuration& c, const std::string& t) = 0;

        virtual void apply(Configuration& c, const std::string& t) = 0;

        virtual const std::string get_oracle(
                Configuration& c,
                DependencyTree& tree) = 0;

        virtual bool is_oracle(
                Configuration& c,
                std::string& t,
                DependencyTree& tree) = 0;

        virtual Configuration init_configuration(
                DependencySent& sent) = 0;

        virtual bool is_terminal(Configuration& c) = 0;

    public:
        std::string lang;
        std::string eval_lang;
        std::string eval_corpora;
        std::string root_label;
        std::vector<std::string> labels;
        std::vector<std::string> transitions;

        bool labeled;
};

#endif
