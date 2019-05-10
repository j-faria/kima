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
		
		/// Prior for the orbital periods.
		std::shared_ptr<DNest4::ContinuousDistribution> Pprior;
		/// Prior for the semi-amplitudes (in m/s).
		std::shared_ptr<DNest4::ContinuousDistribution> Kprior;
		/// Prior for the eccentricities.
		std::shared_ptr<DNest4::ContinuousDistribution> eprior;
		/// Prior for the phases.
		std::shared_ptr<DNest4::ContinuousDistribution> phiprior;
		/// Prior for the .
		std::shared_ptr<DNest4::ContinuousDistribution> wprior;

		// hyperpriors

		/// Prior for the log of the median orbital period
		std::shared_ptr<DNest4::ContinuousDistribution> log_muP_prior;
		/// Prior for the diversity of orbital periods
		std::shared_ptr<DNest4::ContinuousDistribution> wP_prior;
		/// Prior for the log of the mean semi-amplitude
		std::shared_ptr<DNest4::ContinuousDistribution> log_muK_prior;


		/// Generate a point from the prior.
		void from_prior(DNest4::RNG& rng);

		double log_pdf(const std::vector<double>& vec) const;
		/// Get parameter sample from a uniform sample (CDF)
		void from_uniform(std::vector<double>& vec) const;
		/// Get uniform sample from a parameter sample (inverse CDF)
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;

};

#endif

