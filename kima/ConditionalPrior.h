#pragma once

#include <memory>
#include <cmath>
#include <typeinfo>
#include "DNest4.h"


class RVConditionalPrior:public DNest4::ConditionalPrior
{
	private:
		/// whether the model includes hyper-priors for the orbital period and
		/// semi-amplitude
		bool hyperpriors;
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

		// turn on hyperpriors
		void use_hyperpriors();

		/// Prior for the log of the median orbital period
		std::shared_ptr<DNest4::ContinuousDistribution> log_muP_prior;
		/// Prior for the diversity of orbital periods
		std::shared_ptr<DNest4::ContinuousDistribution> wP_prior;
		/// Prior for the log of the mean semi-amplitude
		std::shared_ptr<DNest4::ContinuousDistribution> log_muK_prior;


		/// Generate a point from the prior
		void from_prior(DNest4::RNG& rng);
		/// Get the log prob density at a position `vec`
		double log_pdf(const std::vector<double>& vec) const;
		/// Get parameter sample from a uniform sample (CDF)
		void from_uniform(std::vector<double>& vec) const;
		/// Get uniform sample from a parameter sample (inverse CDF)
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;

};


class BinariesConditionalPrior:public DNest4::ConditionalPrior
{
	private:
		double perturb_hyperparameters(DNest4::RNG& rng);

	public:
		BinariesConditionalPrior();

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
		/// Prior for the linear rate of change of w
		std::shared_ptr<DNest4::ContinuousDistribution> wdotprior;


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



class RVMixtureConditionalPrior:public DNest4::ConditionalPrior
{
	private:
		double tau1, tau2;
		double perturb_hyperparameters(DNest4::RNG& rng);

	public:
		RVMixtureConditionalPrior();

		// priors for all planet parameters

		/// Prior for the mixture probability
		std::shared_ptr<DNest4::ContinuousDistribution> Lprior;
		/// Prior for the orbital periods (in days).
		std::shared_ptr<DNest4::ContinuousDistribution> Pprior;
		/// Prior for the semi-amplitudes (in m/s).
		std::shared_ptr<DNest4::ContinuousDistribution> K1prior;
		std::shared_ptr<DNest4::ContinuousDistribution> K2prior;
		/// Prior for the eccentricities.
		std::shared_ptr<DNest4::ContinuousDistribution> eprior;
		/// Prior for the phases (radians).
		std::shared_ptr<DNest4::ContinuousDistribution> phiprior;
		/// Prior for the argument of periastron (radians).
		std::shared_ptr<DNest4::ContinuousDistribution> wprior;

		/// Priors for the hyperparameters
		// std::shared_ptr<DNest4::ContinuousDistribution> gamma_prior;
		std::shared_ptr<DNest4::ContinuousDistribution> tau1_prior;
		std::shared_ptr<DNest4::ContinuousDistribution> tau2_prior;


		/// Generate a point from the prior
		void from_prior(DNest4::RNG& rng);
		/// Get the log prob density at a position `vec`
		double log_pdf(const std::vector<double>& vec) const;
		/// Get parameter sample from a uniform sample (CDF)
		void from_uniform(std::vector<double>& vec) const override;
		/// Get uniform sample from a parameter sample (inverse CDF)
		void to_uniform(std::vector<double>& vec) const override;

		double bisect_mixture_cdf(double p, double lambda) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;

};



class RVMixtureConditionalPrior1:public DNest4::ConditionalPrior
{
	private:
		double gamma;
		double tau1, tau2;
		double perturb_hyperparameters(DNest4::RNG& rng);

	public:
		RVMixtureConditionalPrior1();

		// priors for all planet parameters

		/// Prior for the orbital periods.
		std::shared_ptr<DNest4::ContinuousDistribution> Pprior;
		/// Prior for the semi-amplitudes (in m/s).
		std::shared_ptr<DNest4::ContinuousDistribution> K1prior;
		std::shared_ptr<DNest4::ContinuousDistribution> K2prior;
		/// Prior for the eccentricities.
		std::shared_ptr<DNest4::ContinuousDistribution> eprior;
		/// Prior for the phases.
		std::shared_ptr<DNest4::ContinuousDistribution> phiprior;
		/// Prior for the .
		std::shared_ptr<DNest4::ContinuousDistribution> wprior;

		/// Priors for the hyperparameters
		std::shared_ptr<DNest4::ContinuousDistribution> gamma_prior;
		std::shared_ptr<DNest4::ContinuousDistribution> tau1_prior;
		std::shared_ptr<DNest4::ContinuousDistribution> tau2_prior;


		/// Generate a point from the prior
		void from_prior(DNest4::RNG& rng);
		/// Get the log prob density at a position `vec`
		double log_pdf(const std::vector<double>& vec) const;
		/// Get parameter sample from a uniform sample (CDF)
		// void from_uniform(DNest4::RNG &rng, std::vector<double>& vec) const;
		void from_uniform(std::vector<double>& vec) const override;
		/// Get uniform sample from a parameter sample (inverse CDF)
		// void to_uniform(DNest4::RNG &rng, std::vector<double>& vec) const;
		void to_uniform(std::vector<double>& vec) const override;

		// double perturb1(DNest4::RNG &rng, const std::vector<std::vector<double>> &components,
		// 				std::vector<std::vector<double>> &u_components);

		// double perturb2(DNest4::RNG &rng, std::vector<std::vector<double>> &components,
		// 				const std::vector<std::vector<double>> &u_components);

		double bisect_mixture_cdf(double p) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;

};
