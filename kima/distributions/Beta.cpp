#include "Beta.h"

namespace DNest4
{

    Beta::Beta(double _a, double _b)
        : a(_a), b(_b)
    {
        if (a <= 0.0 or b <= 0.0)
            throw std::domain_error("Beta distribution must have positive a and b");
    }

    void Beta::setpars(double _a, double _b)
    {
        if (_a <= 0.0 or _b <= 0.0)
            throw std::domain_error("Beta distribution must have positive a and b");
        
        a = _a;
        b = _b;
    }

    double Beta::cdf(double x) const
    {
        if (x < 0.0)
            return 0.0;
        else if (x > 1.0)
            return 1.0;
        return incbeta(a, b, x);
    }

    double Beta::cdf_inverse(double p) const
    {
        if (p < 0.0 || p > 1.0)
            throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        return incomplete_beta_inv(a, b, p);
    }

    double Beta::log_pdf(double x) const
    {
        if (x < 0.0 || x > 1.0)
            return -std::numeric_limits<double>::infinity();

        if (x == 0.0) {
            if (a < 1.0)
                return std::numeric_limits<double>::infinity();
            else if (a > 1.0)
                return -std::numeric_limits<double>::infinity();
            else
                return log(b);
        }

        if (x == 1.0) {
            if (b < 1.0 || b > 1.0)
                return std::numeric_limits<double>::infinity();
            else
                return log(a);
        }

        return -log(std::beta(a, b)) + (a - 1) * log(x) + (b - 1) * log(1 - x);
    }

} // namespace DNest4
