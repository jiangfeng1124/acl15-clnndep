#include "Util.h"
#include <iostream>
#include <cstdlib>
#include "math/mat.h"

using namespace std;

void test_rand()
{
    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < 10; ++j)
            cerr << (Util::rand_double() * 2 - 1) * 0.01;
        cerr << endl;
    }

    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < 10; ++j)
            cerr << (Util::rand_double() * 2 - 1) * 0.01;
        cerr << endl;
    }

    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < 10; ++j)
            cerr << Util::rand_int(1000) % 100;
        cerr << endl;
    }
}

void test_mat_operation()
{
    Mat<double> a(0.8, 4, 4);
    Mat<double> b(1.0, 4, 4);

    cerr << "initialized" << endl;
    Mat<double> c = Util::mat_add(a, b);
    Mat<double> d = Util::mat_subtract(a, b);
    cerr << "calculated" << endl;

    for (int i = 0; i < c.nrows(); ++i)
    {
        for (int j = 0; j < c.ncols(); ++j)
            cerr << c[i][j] << " ";
        cerr << endl;
    }
    for (int i = 0; i < d.nrows(); ++i)
    {
        for (int j = 0; j < d.ncols(); ++j)
            cerr << d[i][j] << " ";
        cerr << endl;
    }

    double n = Util::l2_norm(c);
    cerr << "c.norm = " << n << endl;

    Vec<double> v1(0.7, 5);
    Vec<double> v2(0.6, 5);
    Vec<double> v3 = Util::vec_add(v1, v2);
    Vec<double> v4 = Util::vec_subtract(v1, v2);

    for (int i = 0; i < v3.size(); ++i)
        cerr << v3[i] << " ";
    cerr << endl;
    for (int i = 0; i < v4.size(); ++i)
        cerr << v4[i] << " ";
    cerr << endl;

    cerr << "v3.norm = " << Util::l2_norm(v3) << endl;
    cerr << "v4.norm = " << Util::l2_norm(v4) << endl;
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    test_mat_operation();
    test_rand();

    return 0;
}
