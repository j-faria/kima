#pragma once

#include <vector>
#include <memory>
#include "ConditionalPrior_2.h"
#include "RJObject/RJObject.h"
#include "RNG.h"
#include "DNest4.h"
#include "Data.h"
#include "kepler.h"
#include "AMDstability.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "celerite/celerite.h"

/// whether the model includes a GP component
extern const bool GP;

/// whether the model includes a MA component
extern const bool MA;

/// whether the model includes a linear trend
extern const bool trend;
extern const int degree;

/// whether the data comes from different instruments
/// (and offsets should be included in the model)
extern const bool multi_instrument;

/// include a (better) known extra Keplerian curve? (KO mode!)
extern const bool known_object;
extern const int n_known_object;

/// use a Student-t distribution for the likelihood (instead of Gaussian)
extern const bool studentt;

/// whether to include relativistic corrections
extern const bool relativistic_correction;

class RV_binaries_model
{
    private:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
        int npmax {1};

        DNest4::RJObject<RVConditionalPrior_2> planets =
            DNest4::RJObject<RVConditionalPrior_2>(6, npmax, fix, RVConditionalPrior_2());

        double background;

        std::vector<double> offsets = // between instruments
              std::vector<double>(get_data().number_instruments - 1);
        std::vector<double> jitters = // for each instrument
              std::vector<double>(get_data().number_instruments);

        std::vector<double> betas = // "slopes" for each indicator
              std::vector<double>(get_data().number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double sigmaMA, tauMA;
        double extra_sigma;
        double nu;

        // Parameters for the quasi-periodic extra noise
        enum Kernel {standard, qpc, celerite, permatern32, permatern52, perrq, sqexp};
        std::vector<std::string> _kernels = {"standard", "qpc", "celerite", "permatern32", "permatern52", "perrq", "sqexp"};

        Kernel kernel = standard;

        double eta1, eta2, eta3, eta4, eta5, alpha;
        double log_eta1, log_eta2, log_eta3, log_eta4, log_eta5, log_alpha;
        double a,b,c,P;
        celerite::solver::CholeskySolver<double> solver;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w, KO_wdot;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;
        std::vector<double> KO_wdot;

        // The signal
        std::vector<long double> mu = // the RV model
                            std::vector<long double>(get_data().N());
        void calculate_mu();
        void add_known_object();
        void remove_known_object();
        int is_stable() const;
        bool enforce_stability = false;
        
        double star_mass = 1.0;  // [Msun]
        double binary_mass = 0.0; //if not specified set to zero

        // The covariance matrix for the data
        Eigen::MatrixXd C {get_data().N(), get_data().N()};
        void calculate_C();

        unsigned int staleness;

        void setPriors();
        void save_setup();

    public:
        RV_binaries_model();

        void initialise() {};

        // priors for parameters *not* belonging to the planets
        /// Prior for the systemic velocity.
        std::shared_ptr<DNest4::ContinuousDistribution> Cprior;
        /// Prior for the extra white noise (jitter).
        std::shared_ptr<DNest4::ContinuousDistribution> Jprior;
        /// Prior for the slope (used if `trend = true`).
        std::shared_ptr<DNest4::ContinuousDistribution> slope_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> quadr_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> cubic_prior;
        /// (Common) prior for the between-instruments offsets.
        std::shared_ptr<DNest4::ContinuousDistribution> offsets_prior;
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> individual_offset_prior {
            (size_t) get_data().number_instruments - 1
        };
        /// no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> betaprior;
        /// no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> sigmaMA_prior;
        /// no doc.
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
        /// @brief Prior for the Rational Quadratic shape parameters.
        std::shared_ptr<DNest4::ContinuousDistribution> alpha_prior;
        /// @brief Prior for the ...
        std::shared_ptr<DNest4::ContinuousDistribution> eta5_prior;


        // priors for KO mode!
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Pprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Kprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_eprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_phiprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wdotprior {(size_t) n_known_object};


        std::shared_ptr<DNest4::ContinuousDistribution> nu_prior;

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

        // create an alias for Data::get_instance()
        static RVData& get_data() { return RVData::get_instance(); }

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

