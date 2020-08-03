#include "Gaussian_from_Tc.h"

namespace DNest4
{


Gaussian_from_Tc::Gaussian_from_Tc(double Tc, double errTc, double P, double errP)
    {
        double t0 = Data::get_instance().M0_epoch;

        // propagate uncertainty on Tc and P to phi
        double A, B, AoB, sA, sB;
        A = t0 - Tc;
        B = (2 * M_PI * P);
        AoB = A / B;
        sA = errTc;
        sB = (2 * M_PI * errP);

        // build the Gaussian distribution for phi
        m = AoB;
        s = fabs(AoB) * sqrt(pow(sA/A, 2) + pow(sB/B, 2));
        d = Gaussian(m, s);
    }

    double Gaussian_from_Tc::cdf(double x) const
    {
        return d.cdf(x);
    }

    double Gaussian_from_Tc::cdf_inverse(double p) const
    {
        return d.cdf_inverse(p);
    }

    double Gaussian_from_Tc::log_pdf(double x) const
    {
        return d.log_pdf(x);
    }

    double Gaussian_from_Tc::perturb(double& x, RNG& rng) const
    {
        // (void)x; (void)rng; // to silence the unused parameter warnings
        d.perturb(x, rng);
        return 0.0;
    }


} // namespace DNest4