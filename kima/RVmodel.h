#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "ConditionalPrior.h"
#include "Data.h"
#include "kepler.h"
#include "AMDstability.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace std;
using namespace DNest4;
using namespace kima;
using namespace Eigen;

/// whether the model includes a MA component
extern const bool MA;


/// whether the data comes from different instruments
/// (and offsets should be included in the model)
extern const bool multi_instrument;

/// include a (better) known extra Keplerian curve? (KO mode!)
extern const bool known_object;
extern const int n_known_object;

/// use a Student-t distribution for the likelihood (instead of Gaussian)
extern const bool studentt;

namespace kima {

// template <class conditional=RVConditionalPrior>
class RVmodel
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

        std::vector<double> offsets = // between instruments
              std::vector<double>(data.number_instruments - 1);
        std::vector<double> jitters = // for each instrument
              std::vector<double>(data.number_instruments);

        std::vector<double> betas = // "slopes" for each indicator
              std::vector<double>(data.number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double sigmaMA, tauMA;
        double extra_sigma;
        double nu;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;

        // The signal
        std::vector<double> mu = std::vector<double>(data.N());

        void calculate_mu();
        void add_known_object();
        void remove_known_object();

        double star_mass = 1.0;  // [Msun]
        int is_stable() const;
        bool enforce_stability = false;

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
        /// no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> sigmaMA_prior;
        /// no doc.
        std::shared_ptr<DNest4::ContinuousDistribution> tauMA_prior;

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

        /// Prior for the degrees of freedom $\nu$ of the Student t likelihood
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
        template <class T, class... Args>
        std::shared_ptr<T> make_prior(Args&&... args)
        {
            return std::make_shared<T>(args...);
        }

        // create an alias for RVData::get_instance()
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
