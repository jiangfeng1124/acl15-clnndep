#ifndef __NNDEP_ARCSTANDARD_H__
#define __NNDEP_ARCSTANDARD_H__

#include <vector>
#include "ParsingSystem.h"
#include "Configuration.h"

class ArcStandard : public ParsingSystem
{
    public:
        ArcStandard() { lang = "english"; };
        explicit ArcStandard(std::vector<std::string>& ldict, bool is_labeled = true);
        ~ArcStandard();

        void make_transitions();
        bool can_apply(Configuration& c, const std::string& t);
        void apply(Configuration& c, const std::string& t);
        const std::string get_oracle(
                Configuration& c,
                DependencyTree& tree);
        bool is_oracle(
                Configuration& c,
                std::string& t,
                DependencyTree& tree);
        Configuration init_configuration(DependencySent& sent);
        bool is_terminal(Configuration& c);
};

#endif
