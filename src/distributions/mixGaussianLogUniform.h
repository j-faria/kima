#pragma once

// DNest4/code
#include "DNest4.h"
#include <limits>
#include <iostream>

using namespace std;

namespace DNest4
{

class mixGaussianLogUniform:public ContinuousDistribution
{
    private:
        double mean, sigma, lower, upper;
        DNest4::Gaussian G;
        DNest4::LogUniform LU;

    public:
        mixGaussianLogUniform(double mean = 0.0, double sigma = 1.0,
                              double lower = 1.0, double upper = 2.0);

        double cdf(double x) const;
        double cdf_inverse(double p) const;
        double log_pdf(double x) const;
        // ostream representation of mixGaussianLogUniform class
        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "mixGLU(" << mean << ", " << sigma << ", ";
            out << lower << ", " << upper << ")";
            return out;
        }
        double perturb(double& x, RNG& rng) const;
};


} // namespace DNest4

