#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "Data.h"
#include "utils.h"
#include "ConditionalPrior.h"
#include "kepler.h"
#include "AMDstability.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace std;
using namespace DNest4;
using namespace kima;

namespace kima {

class BINARIESmodel
{
    private:
        RVData data = RVData::get_instance();

        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
        int npmax {1};

        DNest4::RJObject<BinariesConditionalPrior> planets =
            DNest4::RJObject<BinariesConditionalPrior>(6, npmax, fix, BinariesConditionalPrior());

        double bkg, bkg2;

        /// whether the model includes a linear trend
        bool trend {false};
        int degree {0};

        /// use a Student-t distribution for the likelihood (instead of Gaussian)
        bool studentt {false};

        /// include (better) known extra Keplerian curve(s)? (KO mode!)
        bool known_object {true};
        int n_known_object {1};

        /// whether to include relativistic corrections
        bool relativistic_correction {false};

        /// whether to include tidal correction
        bool tidal_correction {false};

        ///whether double-lined binary
        bool double_lined {false};


        //primary
        std::vector<double> offsets = // between instruments
              std::vector<double>(get_data().number_instruments - 1);
        std::vector<double> jitters = // for each instrument
              std::vector<double>(get_data().number_instruments);
              
        //secondary
        std::vector<double> offsets_2 = // between instruments
              std::vector<double>(get_data().number_instruments - 1);
        std::vector<double> jitters_2 = // for each instrument
              std::vector<double>(get_data().number_instruments);
        
        
        std::vector<double> betas = // "slopes" for each indicator
              std::vector<double>(get_data().number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double sigmaMA, tauMA;
        double extra_sigma, extra_sigma_2;
        double nu;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w, KO_wdot;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_q;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;
        std::vector<double> KO_wdot;

        // The signal
        std::vector<double> mu = // the RV model
                            std::vector<double>(data.N());
        std::vector<double> mu_2 = // the RV model for secondary
                            std::vector<double>(data.N()); // changed to imitate RVFWHM get_data replaced by get_instance
        void calculate_mu();
        void calculate_mu_2();
        void add_known_object();
        void add_known_object_secondary();
        void remove_known_object();
        void remove_known_object_secondary();
        int is_stable() const;
        bool enforce_stability = false;
        
        double star_mass = 1.0;  // [Msun]
        double star_radius = 0.0; //if not specified set to zero
        double binary_mass = 0.0; //if not specified set to zero

        // The covariance matrix for the data
        Eigen::MatrixXd C {get_data().N(), get_data().N()};
        void calculate_C();

        unsigned int staleness;

        void setPriors();
        void save_setup();

    public:
        BINARIESmodel();

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
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_qprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_eprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_phiprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wdotprior {(size_t) n_known_object};

        std::shared_ptr<DNest4::ContinuousDistribution> nu_prior;

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

}  // namespace kima
