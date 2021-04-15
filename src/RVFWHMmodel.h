#pragma once

#include <vector>
#include <memory>
#include "RVConditionalPrior.h"
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

class RVFWHMmodel
{
    private:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
        int npmax {1};

        DNest4::RJObject<RVConditionalPrior> planets =
            DNest4::RJObject<RVConditionalPrior>(5, npmax, fix, RVConditionalPrior());

        double bkg, bkg2;

        std::vector<double> offsets = // between instruments
              std::vector<double>(2*Data::get_instance().number_instruments - 2);
        std::vector<double> jitters = // for each instrument
              std::vector<double>(2*Data::get_instance().number_instruments);

        std::vector<double> betas = // "slopes" for each indicator
              std::vector<double>(Data::get_instance().number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double sigmaMA, tauMA;
        double jitter1, jitter2;
        double nu;

        // Parameters for the quasi-periodic extra noise
        enum Kernel {standard, qpc, celerite};
        std::vector<std::string> _kernels = {"standard", "qpc", "celerite"};
        Kernel kernel = standard;

        celerite::solver::CholeskySolver<double> solver;

        // hyper parameters for RV (1st output)
        double eta1_1, eta2_1, eta3_1, eta4_1, eta5_1;
        double log_eta1_1, log_eta2_1, log_eta3_1, log_eta4_1, log_eta5_1;
        // hyper parameters for FWHM (2nd output)
        double eta1_2, eta2_2, eta3_2, eta4_2, eta5_2;
        double log_eta1_2, log_eta2_2, log_eta3_2, log_eta4_2, log_eta5_2;

        // share some hyperparameters?
        bool share_eta2 {true};
        bool share_eta3 {true};
        bool share_eta4 {true};
        bool share_eta5 {false};

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;

        // The signals
        std::vector<long double> mu = // the RV model
                            std::vector<long double>(Data::get_instance().N());
        std::vector<long double> mu_2 = // the 2nd output model
                            std::vector<long double>(Data::get_instance().N());

        void calculate_mu();
        void calculate_mu_2();
        void add_known_object();
        void remove_known_object();
        int is_stable() const;
        bool enforce_stability = false;

        double star_mass = 1.0;  // [Msun]

        // eccentric and true anomalies
        double ecc_anomaly(double time, double prd, double ecc, double peri_pass);
        double eps3(double e, double M, double x);
        double keplerstart3(double e, double M);
        double true_anomaly(double time, double prd, double ecc, double peri_pass);

        // The covariance matrices for the data
        Eigen::MatrixXd C_1 {Data::get_instance().N(), Data::get_instance().N()};
        Eigen::MatrixXd C_2 {Data::get_instance().N(), Data::get_instance().N()};
        void calculate_C_1();
        void calculate_C_2();

        unsigned int staleness;

        void setPriors();
        void save_setup();

    public:
        RVFWHMmodel();

        // priors for parameters *not* belonging to the planets
        /// Prior for the systemic velocity.
        std::shared_ptr<DNest4::ContinuousDistribution> Vprior;
        std::shared_ptr<DNest4::ContinuousDistribution> C2prior;
        /// Prior for the extra white noise (jitter).
        std::shared_ptr<DNest4::ContinuousDistribution> Jprior;
        std::shared_ptr<DNest4::ContinuousDistribution> J2prior;
        /// Prior for the slope (used if `trend = true`).
        std::shared_ptr<DNest4::ContinuousDistribution> slope_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> quadr_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> cubic_prior;
        /// (Common) prior for the between-instruments offsets.
        std::shared_ptr<DNest4::ContinuousDistribution> offsets_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> offsets2_prior;
        /// no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> betaprior;
        /// no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> sigmaMA_prior;
        /// no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> tauMA_prior;


        // priors for the hyperparameters

        /// Prior for eta1, the GP variance.
        std::shared_ptr<DNest4::ContinuousDistribution> eta1_1_prior;
        /// Prior for eta2, the GP correlation timescale.
        std::shared_ptr<DNest4::ContinuousDistribution> eta2_1_prior;
        /// Prior for eta3, the GP period.
        std::shared_ptr<DNest4::ContinuousDistribution> eta3_1_prior;
        /// Prior for eta4, the recurrence timescale.
        std::shared_ptr<DNest4::ContinuousDistribution> eta4_1_prior;
        /// Prior for eta5, ...
        std::shared_ptr<DNest4::ContinuousDistribution> eta5_1_prior;

        // same for the FWHM
        std::shared_ptr<DNest4::ContinuousDistribution> eta1_2_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> eta2_2_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> eta3_2_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> eta4_2_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> eta5_2_prior;


        // priors for KO mode!
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Pprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Kprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_eprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_phiprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wprior {(size_t) n_known_object};



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

