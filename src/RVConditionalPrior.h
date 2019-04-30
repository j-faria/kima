// (c) 2019 João Faria
// This file is part of kima, which is licensed under the MIT license (see LICENSE for details)

#ifndef DNest4_RVConditionalPrior
#define DNest4_RVConditionalPrior

#include <memory>
#include "RNG.h"
#include "RJObject/ConditionalPriors/ConditionalPrior.h"
#include "DNest4.h"

/// whether the model includes hyper-priors for the orbital period and semi-amplitude
extern const bool hyperpriors;

class RVConditionalPrior:public DNest4::ConditionalPrior
{
	private:
		// Parameters of bi-exponential hyper-distribution for log-periods
		double center, width;

		// Mean of exponential hyper-distribution for semi-amplitudes
		double muK;

		double perturb_hyperparameters(DNest4::RNG& rng);

	public:
		RVConditionalPrior();

		// priors for all planet parameters
		/// @brief Prior for the orbital periods.
		std::shared_ptr<DNest4::ContinuousDistribution> Pprior;
		/// @brief Prior for the semi-amplitudes (in m/s).
		std::shared_ptr<DNest4::ContinuousDistribution> Kprior;
		/// @brief Prior for the eccentricities.
		std::shared_ptr<DNest4::ContinuousDistribution> eprior;
		/// @brief Prior for the phases.
		std::shared_ptr<DNest4::ContinuousDistribution> phiprior;
		/// @brief Prior for the .
		std::shared_ptr<DNest4::ContinuousDistribution> wprior;

		/// @brief Generate a point from the prior. 
		void from_prior(DNest4::RNG& rng);

		double log_pdf(const std::vector<double>& vec) const;
		void from_uniform(std::vector<double>& vec) const;
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;

};

#endif

