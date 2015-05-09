#ifndef __NNDEP_DEPENDENCY_SENT_H__
#define __NNDEP_DEPENDENCY_SENT_H__

#include <vector>
#include <string>

class DependencySent
{
    public:
        DependencySent();
        DependencySent(const DependencySent& s);
        ~DependencySent() {}

        void add(std::string& word, std::string& pos, std::string& cluster);

        void init();

        void print_info();

    public:
        int n;
        std::vector<std::string> words;
        std::vector<std::string> poss;
        std::vector<std::string> clusters;
        // std::vector<std::string> pposs; #TODO
};

#endif
