#pragma once

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

/// use a Student-t distribution for the likelihood (instead of Gaussian)
extern const bool studentt;

class RVmodel
{
    private:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
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

        double slope, quadr=0.0, cubic=0.0;
        double sigmaMA, tauMA;
        double extra_sigma;
        double nu;

        // Parameters for the quasi-periodic extra noise
        enum Kernel {standard, celerite};
        Kernel kernel = standard;
        double eta1, eta2, eta3, eta4, eta5;
        double log_eta1, log_eta2, log_eta3, log_eta4, log_eta5;
        double a,b,c,P;
        celerite::solver::CholeskySolver<double> solver;

        // Parameters for the known object, if set
        double KO_P, KO_K, KO_e, KO_phi, KO_w;

        // The signal
        std::vector<long double> mu = // the RV model
                            std::vector<long double>(Data::get_instance().N());
        void calculate_mu();
        void add_known_object();
        void remove_known_object();

        // eccentric and true anomalies
        double ecc_anomaly(double time, double prd, double ecc, double peri_pass);
        double eps3(double e, double M, double x);
        double keplerstart3(double e, double M);
        double true_anomaly(double time, double prd, double ecc, double peri_pass);

        template <typename T> inline void sin_cos_reduc (T x, T* SnReduc, T* CsReduc);
        template <typename T> inline T solve_kepler (T t, T period, T ecc, T time_peri);

        // The covariance matrix for the data
        Eigen::MatrixXd C {Data::get_instance().N(), Data::get_instance().N()};
        void calculate_C();

        unsigned int staleness;

        void setPriors();
        void save_setup();

    public:
        RVmodel();

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


        // priors for KO mode!
        std::shared_ptr<DNest4::ContinuousDistribution> KO_Pprior;
        std::shared_ptr<DNest4::ContinuousDistribution> KO_Kprior;
        std::shared_ptr<DNest4::ContinuousDistribution> KO_eprior;
        std::shared_ptr<DNest4::ContinuousDistribution> KO_phiprior;
        std::shared_ptr<DNest4::ContinuousDistribution> KO_wprior;

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
        Data& get_data() { return Data::get_instance(); }

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

