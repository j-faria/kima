#pragma once

#include <stdexcept>
#include <cmath>
#include "Distributions/ContinuousDistribution.h"
#include "RNG.h"
#include "incbeta.h"
#include "invincbeta.h"

namespace DNest4
{

    /*
     * Beta distribution
     * https://en.wikipedia.org/wiki/Beta_distribution
     */
    class Beta : public ContinuousDistribution
    {
    private:
        double a, b;

    public:
        Beta(double a = 1.0, double b = 1.0);
        void setpars(double a, double b);

        double cdf(double x) const;
        double cdf_inverse(double p) const;
        double log_pdf(double x) const;
        // ostream representation of Beta class
        virtual std::ostream &print(std::ostream &out) const override
        {
            out << "Beta(" << a << "; " << b << ")";
            return out;
        }
    };

} // namespace DNest4

