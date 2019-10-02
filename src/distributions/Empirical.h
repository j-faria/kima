#ifndef DNest4_Empirical
#define DNest4_Empirical

// DNest4/code
#include "Distributions/ContinuousDistribution.h"
#include <limits>
#include <stdlib.h>
#include <stdint.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector> 
#include <algorithm>

using namespace std;

namespace DNest4
{

class Empirical:public ContinuousDistribution
{
    private:
        vector<double> data = {};

    public:
        Empirical(const char* filename);
        Empirical(const vector<double> data);

        double cdf(double x) const;
        double cdf_inverse(double p) const;
        double log_pdf(double x) const;
};


} // namespace DNest4

#endif

