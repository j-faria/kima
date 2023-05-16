#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "utils.h"
#include "kepler.h"
#include "AMDstability.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace std;
using namespace DNest4;
using namespace kima;


namespace kima {

// template <class conditional=RVConditionalPrior>
class GPmodel
{
    private:
        RVData data = RVData::get_instance();

        /// Fix the number of planets? (by default, yes)
        bool fix {true};

        /// Maximum number of planets (by default 1)
        int npmax {1};

        DNest4::RJObject<RVConditionalPrior> planets =
            DNest4::RJObject<RVConditionalPrior>(5, npmax, fix, RVConditionalPrior());

        double background;

        /// whether the model includes a linear trend
        bool trend {false};
        int degree {0};

        /// include (better) known extra Keplerian curve(s)? (KO mode!)
        bool known_object {false};
        int n_known_object {0};

        std::vector<double> offsets = // between instruments
              std::vector<double>(data.number_instruments - 1);
        std::vector<double> jitters = // for each instrument
              std::vector<double>(data.number_instruments);

        std::vector<double> betas = // "slopes" for each indicator
              std::vector<double>(data.number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double extra_sigma;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;

        // Parameters for the Gaussian process
        enum Kernel
        {
            standard,
            qpc,
            celerite,
            permatern32,
            permatern52,
            perrq,
            sqexp,
            periodic
        };
        std::vector<std::string> _kernels = {
            "standard",
            "qpc",
            "celerite",
            "permatern32",
            "permatern52",
            "perrq",
            "sqexp",
            "periodic"
        };

        Kernel kernel = standard;

        double eta1, eta2, eta3, eta4, eta5, alpha;
        double log_eta1, log_eta2, log_eta3, log_eta4, log_eta5, log_alpha;

        // The signal
        std::vector<double> mu = std::vector<double>(data.N());
        // The covariance matrix for the data
        Eigen::MatrixXd C {data.N(), data.N()};

        void calculate_mu();
        void add_known_object();
        void remove_known_object();

        double star_mass = 1.0;  // [Msun]
        int is_stable() const;
        bool enforce_stability = false;

        bool hyperpriors {false}; // defined by the conditional

        unsigned int staleness;


    public:
        GPmodel();

        // priors for parameters *not* belonging to the planets
        /// Prior for the systemic velocity.
        std::shared_ptr<DNest4::ContinuousDistribution> Cprior;
        /// Prior for the extra white noise (jitter).
        std::shared_ptr<DNest4::ContinuousDistribution> Jprior;
        /// Prior for the slope
        std::shared_ptr<DNest4::ContinuousDistribution> slope_prior;
        /// Prior for the quadratic coefficient of the trend
        std::shared_ptr<DNest4::ContinuousDistribution> quadr_prior;
        /// Prior for the cubic coefficient of the trend
        std::shared_ptr<DNest4::ContinuousDistribution> cubic_prior;
        /// (Common) prior for the between-instruments offsets.
        std::shared_ptr<DNest4::ContinuousDistribution> offsets_prior;
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> individual_offset_prior {
            (size_t) data.number_instruments - 1
        };
        /// no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> betaprior;

        // priors for KO mode!
        /// Prior for the KO orbital period(s)
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Pprior {(size_t) n_known_object};
        /// Prior for the KO semi-amplitude(s)
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Kprior {(size_t) n_known_object};
        /// Prior for the KO eccentricity(ies)
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_eprior {(size_t) n_known_object};
        /// Prior for the KO mean anomaly(ies)
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_phiprior {(size_t) n_known_object};
        /// Prior for the KO argument(s) of pericenter
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wprior {(size_t) n_known_object};

        // priors for the GP hyperparameters
        /// Prior for $\eta_1$, the GP "amplitude"
        std::shared_ptr<DNest4::ContinuousDistribution> eta1_prior;
        /// Prior for $\eta_2$, the GP correlation timescale
        std::shared_ptr<DNest4::ContinuousDistribution> eta2_prior;
        /// Prior for $\eta_3$, the GP period
        std::shared_ptr<DNest4::ContinuousDistribution> eta3_prior;
        /// Prior for $\eta_4$, the recurrence timescale
        std::shared_ptr<DNest4::ContinuousDistribution> eta4_prior;
        /// Prior for the Rational Quadratic shape parameter
        std::shared_ptr<DNest4::ContinuousDistribution> alpha_prior;
        /// Prior for the "amplitude" of the cosine term of the QPC kernel
        std::shared_ptr<DNest4::ContinuousDistribution> eta5_prior;

        /// @brief an alias for RVData::get_instance()
        static RVData& get_data() { return RVData::get_instance(); }

        /// @brief Generate a point from the prior.
        void from_prior(DNest4::RNG& rng);

        /// @brief Set the default priors
        void setPriors();

        /// @brief Save the setup of this model
        void save_setup();

        /// @brief Do Metropolis-Hastings proposals.
        double perturb(DNest4::RNG& rng);

        /// @brief Build the covariance matrix
        void calculate_C();

        /// @brief log-likelihood function
        double log_likelihood() const;

        // Print parameters to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;

        void known_object_mode(int n);

};


}  // namespace kima
