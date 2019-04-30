#ifndef DNest4_RVmodel
#define DNest4_RVmodel

#include <vector>
#include <memory>
#include "RVConditionalPrior.h"
#include "RJObject/RJObject.h"
#include "RNG.h"
#include "Data.h"
#include "DNest4.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

/// whether the model includes a GP component
extern const bool GP;

/// whether the model includes a MA component
extern const bool MA;

/// whether there are observations after the change in HARPS fibers
extern const bool obs_after_HARPS_fibers;

/// whether the model includes a linear trend
extern const bool trend;

/// whether the data comes from different instruments
/// (and offsets should be included in the model)
extern const bool multi_instrument;


class RVmodel
{
    private:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets
        int npmax {1};

        DNest4::RJObject<RVConditionalPrior> planets =
            DNest4::RJObject<RVConditionalPrior>(5, npmax, fix, RVConditionalPrior());

        double background;

        std::vector<double> offsets = // between instruments
              std::vector<double>(Data::get_instance().number_instruments - 1);
        std::vector<double> jitters = // for each instrument
              std::vector<double>(Data::get_instance().number_instruments);

        std::vector<double> betas = // "slopes" for each indicator
              std::vector<double>(Data::get_instance().number_indicators);

        double slope, quad;
        double fiber_offset;
        double sigmaMA, tauMA;
        double extra_sigma;

        // Parameters for the quasi-periodic extra noise
        double eta1, eta2, eta3, eta4, eta5;
        double log_eta1, log_eta2, log_eta3, log_eta4, log_eta5;
        double a,b,c,P;

        // The signal
        std::vector<long double> mu = // the RV model
                            std::vector<long double>(Data::get_instance().N());
        void calculate_mu();

        // eccentric and true anomalies
        double ecc_anomaly(double time, double prd, double ecc, double peri_pass);
        double eps3(double e, double M, double x);
        double keplerstart3(double e, double M);
        double true_anomaly(double time, double prd, double ecc, double peri_pass);

        // The covariance matrix for the data
        Eigen::MatrixXd C {Data::get_instance().N(), Data::get_instance().N()};
        void calculate_C();

        unsigned int staleness;

        void setPriors();

    public:
        RVmodel();

        // priors for parameters *not* belonging to the planets
        /// @brief Prior for the systemic velocity.
        std::shared_ptr<DNest4::ContinuousDistribution> Cprior;
        /// @brief Prior for the extra white noise (jitter).
        std::shared_ptr<DNest4::ContinuousDistribution> Jprior;
        /// @brief Prior for the slope (used if `trend = true`).
        std::shared_ptr<DNest4::ContinuousDistribution> slope_prior;
        /// @brief Prior for the HARPS fiber RV offset.
        std::shared_ptr<DNest4::ContinuousDistribution> fiber_offset_prior;
        /// @brief (Common) prior for the between-instruments offsets.
        std::shared_ptr<DNest4::ContinuousDistribution> offsets_prior;
        /// @brief no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> betaprior;
        /// @brief no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> sigmaMA_prior;
        /// @brief no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> tauMA_prior;

        // priors for the hyperparameters
        /// @brief Prior for the log of eta1, the GP variance.
        std::shared_ptr<DNest4::ContinuousDistribution> log_eta1_prior;
        // std::shared_ptr<DNest4::ContinuousDistribution> log_eta2_prior;
        /// @brief Prior for eta2, the GP correlation timescale.
        std::shared_ptr<DNest4::ContinuousDistribution> eta2_prior;
        /// @brief Prior for eta3, the GP period.
        std::shared_ptr<DNest4::ContinuousDistribution> eta3_prior;
        /// @brief Prior for the log of eta4, the recurrence timescale.
        std::shared_ptr<DNest4::ContinuousDistribution> log_eta4_prior;


        // change the name of std::make_shared :)
        /**
         * @brief Assign a prior distribution.
         * 
         * This function defines, initializes, and assigns a prior distribution.
         * Possible distributions are ...
         * 
         * For example:
         * 
         * @code{.cpp}
         *          Cprior = make_prior<Uniform>(0, 1);
         * @endcode
         * 
         * @tparam T     ContinuousDistribution
         * @tparam Args  
         * @param args   Arguments for constructor of distribution
         * @return std::shared_ptr<T> 
        */
        template< class T, class... Args >
        std::shared_ptr<T> make_prior( Args&&... args ) { return std::make_shared<T>(args...); }

        /// @brief Generate a point from the prior.
        void from_prior(DNest4::RNG& rng);

        /// @brief Do Metropolis-Hastings proposals.
        double perturb(DNest4::RNG& rng);

        // Likelihood function
        double log_likelihood() const;

        // Print parameters to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;

};

#endif

