#pragma once

#include "../src/Data.h"
// DNest4/code
#include "Distributions/ContinuousDistribution.h"
#include "Distributions/Gaussian.h"
#include "RNG.h"

namespace DNest4
{
    /// This represents a Gaussian distribution for phi (the mean anomaly at the
    /// epoch) built from the time of transit Tc and the orbital period P.
    /// kima uses phi as a parameter, but most often the transit provides
    /// information on Tc and P, including their uncertainties.
    class Gaussian_from_Tc:public ContinuousDistribution
    {
        private:
            double m, s;
            Gaussian d;

        public:

            /**
             * @brief Gaussian distribution for phi, given Tc and P.
             * 
             * For example:
             * 
             * @code{.cpp}
             *          phiprior = make_prior<Gaussian_from_Tc>(57000, 0.1, 20, 0.1);
             * @endcode
             * 
             * @param Tc     time of transit
             * @param errTc  uncertainty on the time of transit
             * @param P      orbital period
             * @param errP   uncertainty on the orbital period
            */
            Gaussian_from_Tc(double Tc, double errTc, double P, double errP);

            double cdf(double x) const;
            double cdf_inverse(double p) const;
            double log_pdf(double x) const;
            // ostream representation of Gaussian_from_Tc class
            virtual std::ostream& print(std::ostream& out) const override
            {
                out << "Gaussian(" << m << ", " << s << ")";
                return out;
            }
            // this special class reimplements perturb
            double perturb(double& x, RNG& rng) const;
    };


} // namespace DNest4