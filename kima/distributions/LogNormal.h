#pragma once

#include <stdexcept>
#include <cmath>
#include "Distributions/ContinuousDistribution.h"
#include "RNG.h"
#include "Utils.h"

namespace DNest4
{

    /*
     * Log-normal distribution
     * https://en.wikipedia.org/wiki/Log-normal_distribution
     */
    class LogNormal : public ContinuousDistribution
    {
    private:
        double mu, sigma;

    public:
        LogNormal(double mu=0.0, double sigma=1.0);
        void setpars(double mu, double sigma);

        double cdf(double x) const;
        double cdf_inverse(double p) const;
        double log_pdf(double x) const;
        // ostream representation of LogNormal class
        virtual std::ostream &print(std::ostream &out) const override
        {
            out << "LogNormal(" << mu << "; " << sigma << ")";
            return out;
        }
    };

    /*
     * Truncated Log-normal distribution
     */
    class TruncatedLogNormal : public ContinuousDistribution
    {
    private:
        double mu, sigma;
        double lower, upper; // truncation bounds
        LogNormal unLN; // the original, untruncated, LogNormal distribution
        double c;

    public:
        TruncatedLogNormal(double mu=0.0, double sigma=1.0, 
                           double lower=0.0, double upper=1. / 0.);
        void setpars(double mu, double sigma, double lower, double upper);
        void setpars(double mu, double sigma);

        double cdf(double x) const;
        double cdf_inverse(double p) const;
        double log_pdf(double x) const;
        // ostream representation of TruncatedLogNormal class
        virtual std::ostream &print(std::ostream &out) const override
        {
            out << "TruncatedLogNormal(" << mu << "; " << sigma << "; [" << lower << " , " << upper << "])";
            return out;
        }
    };


} // namespace DNest4

