#include "LogUniform.h"
#include <stdexcept>
#include <cmath>
#include "../Utils.h"

namespace DNest4
{
    /**************************************************************************/
    LogUniform::LogUniform(double _lower, double _upper)
        : lower(_lower), upper(_upper)
    {
        if (lower <= 0.0)
            throw std::domain_error("LogUniform distribution must have positive lower limit.");
        if (upper <= lower)
            throw std::domain_error("LogUniform distribution must have lower < upper limits");

        d = log(upper / lower);
    }

    void LogUniform::setpars(double lo, double up)
    {
        if (lo <= 0.0)
            throw std::domain_error("LogUniform distribution must have positive lower limit");
        if (up <= lo)
            throw std::domain_error("LogUniform distribution must have lower < upper limits");

        lower = lo;
        upper = up;
        d = log(upper / lower);
    }

    double LogUniform::cdf(double x) const
    {
        if (x < lower)
            return 0.0;
        if (x > upper)
            return 1.0;
        return (log(x) - log(lower)) / d;
    }

    double LogUniform::cdf_inverse(double x) const
    {
        if (x < 0.0 || x > 1.0)
            throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        return lower * pow(upper / lower, x);
    }

    double LogUniform::log_pdf(double x) const
    {
        if (x < lower || x > upper)
            return -std::numeric_limits<double>::infinity();
        return -log(x) - log(d);
    }

    /**************************************************************************/
    ModifiedLogUniform::ModifiedLogUniform(double _knee, double _upper)
        : knee(_knee), upper(_upper)
    {
        if (knee <= 0.0)
            throw std::domain_error("ModifiedLogUniform distribution must have positive knee.");
        if (upper <= knee)
            throw std::domain_error("ModifiedLogUniform distribution must have knee < upper limit");

        d = 1.0 / log(1.0 + upper / knee);
    }

    void ModifiedLogUniform::setpars(double kn, double up)
    {
        if (kn <= 0.0)
            throw std::domain_error("ModifiedLogUniform distribution must have positive knee");
        if (up <= kn)
            throw std::domain_error("ModifiedLogUniform distribution must have knee < upper limit");

        knee = kn;
        upper = up;
        d = 1.0 / log(1.0 + upper / knee);
    }

    double ModifiedLogUniform::cdf(double x) const
    {
        return d * log(1.0 + x / knee);
    }

    double ModifiedLogUniform::cdf_inverse(double x) const
    {
        if (x < 0.0 || x > 1.0)
            throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        return (exp(x / d) - 1.0) * knee;
    }

    double ModifiedLogUniform::log_pdf(double x) const
    {
        return log(d) - log(1.0 + x / knee);
    }

} // namespace DNest4
