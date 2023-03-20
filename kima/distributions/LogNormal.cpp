#include "LogNormal.h"

namespace DNest4
{

    LogNormal::LogNormal(double _mu, double _sigma)
        : mu(_mu), sigma(_sigma)
    {
        if (sigma <= 0.0)
            throw std::domain_error("LogNormal distribution must have positive width");
    }

    void LogNormal::setpars(double _mu, double _sigma)
    {
        if (_sigma <= 0.0)
            throw std::domain_error("LogNormal distribution must have positive width");
        
        mu = _mu;
        sigma = _sigma;
    }

    double LogNormal::cdf(double x) const
    {
        if (x <= 0.0)
            return 0.0;
        return normal_cdf((log(x) - mu) / sigma);
    }

    double LogNormal::cdf_inverse(double p) const
    {
        if (p < 0.0 || p > 1.0)
            throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        return exp(mu + sigma * normal_inverse_cdf(p));
    }

    double LogNormal::log_pdf(double x) const
    {
        if (x < 0.0)
            return -std::numeric_limits<double>::infinity();

        if (x == 0.0)
            return -std::numeric_limits<double>::infinity();

        double r = x - mu;
        return - pow(log(r), 2) / (2*pow(sigma, 2)) - log(sigma * r * sqrt(2*M_PI));
    }

    /**************************************************************************/

    TruncatedLogNormal::TruncatedLogNormal(double m, double s, double lo, double up)
        : mu(m), sigma(s), lower(lo), upper(up)
    {
        if (sigma <= 0.0)
            throw std::domain_error("TruncatedLogNormal distribution must have positive width");
        if (lower >= upper)
            throw std::domain_error("TruncatedLogNormal: lower bound should be less than upper bound");

        unLN = LogNormal(mu, sigma);
        c = unLN.cdf(upper) - unLN.cdf(lower);
    }

    void TruncatedLogNormal::setpars(double m, double s, double lo, double up)
    {
        if (s <= 0.0)
            throw std::domain_error("TruncatedLogNormal distribution must have positive width");
        if (lo >= up)
            throw std::domain_error("TruncatedLogNormal: lower bound should be less than upper bound");

        mu = m;
        sigma = s;
        unLN.setpars(mu, sigma);
        lower = lo;
        upper = up;
        c = unLN.cdf(upper) - unLN.cdf(lower);
    }

    void TruncatedLogNormal::setpars(double m, double s)
    {
        if (s <= 0.0)
            throw std::domain_error("TruncatedLogNormal distribution must have positive width");

        mu = m;
        sigma = s;
        unLN.setpars(mu, sigma);
        c = unLN.cdf(upper) - unLN.cdf(lower);
    }

    double TruncatedLogNormal::cdf(double x) const
    {
        double up = std::max(std::min(x, upper), lower);
        return (unLN.cdf(up) - unLN.cdf(lower)) / c;
    }

    double TruncatedLogNormal::cdf_inverse(double x) const
    {
        if (x < 0.0 || x > 1.0)
            throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        double xx = unLN.cdf(lower) + x * c;
        return unLN.cdf_inverse(xx);
    }

    double TruncatedLogNormal::log_pdf(double x) const
    {
        if (x < lower or x > upper)
            return -std::numeric_limits<double>::infinity();
        return unLN.log_pdf(x) - log(c);
    }



} // namespace DNest4
