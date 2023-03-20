#include "BetaBinom.h"

namespace DNest4
{

    BetaBinom::BetaBinom(int n, double a, double b) : n(n), a(a), b(b)
    {
        if (n < 0)
            throw std::domain_error("BetaBinom distribution must have n >= 0");
        if (a <= 0.0 or b <= 0.0)
            throw std::domain_error("BetaBinom distribution must have positive a and b");
    }

    void BetaBinom::setpars(int _n, double _a, double _b)
    {
        if (n < 0)
            throw std::domain_error("setpars:BetaBinom distribution must have n >= 0");
        if (_a <= 0.0 or _b <= 0.0)
            throw std::domain_error("setpars:BetaBinom distribution must have positive a and b");

        n = _n;        
        a = _a;
        b = _b;
    }

    double BetaBinom::cdf(double x) const
    {
        if (x < 0.0)
            return 0.0;
        else if (x >= n)
            return 1.0;
        double sum = 0.0;
        for (double i = 0.0; i <= floor(x); i++)
            sum += exp(log_pdf(i));
        return sum;
    }

    double BetaBinom::bisect_cdf(double p) const
    {
        double a = -1;
        double b = n;
        double c = a;
        while ((b - a) >= 0.1)
        {
            // Find middle point
            c = 0.5 * (a + b);
            double cdf_c = cdf(c);
            double cdf_a = cdf(a);
            if (cdf_c - p == 0.0)
                break;
            else if ((cdf_c - p) * (cdf_a - p) < 0.0)
                b = c;
            else
                a = c;
        }
        return c;
    }

    double BetaBinom::cdf_inverse(double p) const
    {
        if (p < 0.0 || p > 1.0)
            throw std::domain_error("BetaBinom: input to cdf_inverse must be in [0, 1].");
        return std::abs(std::round(bisect_cdf(p)));
    }

    double BetaBinom::log_pdf(double x) const
    {
        if (x < 0.0)
            return -std::numeric_limits<double>::infinity();

        double k = floor(x);
        double combiln = -log(n + 1) - log(std::beta(n - k + 1, k + 1));
        return combiln + log(std::beta(k + a, n - k + b)) - log(std::beta(a, b));
    }

} // namespace DNest4
