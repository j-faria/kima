#pragma once

#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

// DNest4/code
#include "Distributions/ContinuousDistribution.h"
#include "RNG.h"

// stats
#include "stats.hpp"

// #include <boost/math/distributions/inverse_gamma.hpp>

namespace DNest4
{

/*
* Inverse Gamma distribution
*/
class InvGamma:public ContinuousDistribution
{
    private:
        double shape, scale; // shape parameter and scale parameter

    public:
        InvGamma(double shape, double scale);
        void setpars(double shape, double scale);
        // std::tuple<double, double> support() { return std::make_tuple(0, 1./0.); };

        double cdf(double x) const;
        double cdf_inverse(double p) const;
        double log_pdf(double x) const;
        // ostream representation of InvGamma class
        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "InvGamma(" << shape << "; " << scale << ")";
            return out;
        }
};

} // namespace DNest4


