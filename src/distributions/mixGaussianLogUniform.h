#pragma once

// DNest4/code
#include "DNest4.h"
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

class mixGaussianLogUniform:public ContinuousDistribution
{
    private:
        DNest4::Gaussian G;
        DNest4::LogUniform LU;

    public:
        mixGaussianLogUniform(double mean=0.0, double sigma=1.0, double lower=1.0, double upper=2.0);

        double cdf(double x) const;
        double cdf_inverse(double p) const;
        double log_pdf(double x) const;
        double perturb(double& x, RNG& rng) const override;
};


} // namespace DNest4

