#pragma once

#include <stdexcept>
#include <cmath>
#include "Distributions/ContinuousDistribution.h"
#include "RNG.h"
// #include "incbeta.h"
// #include "invincbeta.h"

namespace DNest4
{

    /*
     * Beta-binomial distribution
     * https://en.wikipedia.org/wiki/Beta-binomial_distribution
     */
    class BetaBinom : public ContinuousDistribution
    {
    private:
        int n;
        double a, b;

    public:
        BetaBinom(int n = 1, double a = 1.0, double b = 1.0);
        void setpars(int n, double a, double b);

        double cdf(double x) const;
        double cdf_inverse(double p) const;
        double log_pdf(double x) const;
        // ostream representation of BetaBinom class
        virtual std::ostream &print(std::ostream &out) const override
        {
            out << "BetaBinom(" << n << "; " << a << "; " << b << ")";
            return out;
        }

        double bisect_cdf(double p) const;

    };

} // namespace DNest4

