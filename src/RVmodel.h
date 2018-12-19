#ifndef DNest4_RVmodel
#define DNest4_RVmodel

#include <vector>
#include "RVConditionalPrior.h"
#include "RJObject/RJObject.h"
#include "RNG.h"
#include "Data.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "GPRN.h"

/* whether the model includes a GP component */
extern const bool GP;

/* whether the model is to be a GPRN */
extern const bool RN;

/* whether there are observations after the change in HARPS fibers */
extern const bool obs_after_HARPS_fibers;

/* whether the model includes a linear trend */
extern const bool trend;


class RVmodel
{
    private:
        /* Fix the number of planets? (by default, yes) */
        bool fix {true};
        /* Maximum number of planets */
        int npmax {1};

        DNest4::RJObject<RVConditionalPrior> planets =
            DNest4::RJObject<RVConditionalPrior>(5, npmax, fix, RVConditionalPrior());

        double background;
        double slope, quad;
        double fiber_offset;

        /* Parameters for the quasi-periodic extra noise */
        double eta1, eta2, eta3, eta4, eta5;
        double log_eta1, log_eta2, log_eta3, log_eta4, log_eta5;
        double a,b,c,P;

        /*celerite::solver::CholeskySolver<double> solver;
        Eigen::VectorXd alpha_real,
                 beta_real,
                 alpha_complex_real,
                 alpha_complex_imag,
                 beta_complex_real,
                 beta_complex_imag;*/

        /* The signal */
        std::vector<long double> mu = 
                            std::vector<long double>(Data::get_instance().N());
        void calculate_mu();

        /* eccentric and true anomalies */
        double ecc_anomaly(double time, double prd, double ecc, double peri_pass);
        double eps3(double e, double M, double x);
        double keplerstart3(double e, double M);
        double true_anomaly(double time, double prd, double ecc, double peri_pass);

        /* The covariance matrix for the data */
        Eigen::MatrixXd C {Data::get_instance().N(), Data::get_instance().N()};
        void calculate_C();
        std::vector<Eigen::MatrixXd> Cs {4};

        /* GPRN priors */
        int n_size = GPRN::get_instance().node.size(); //number of nodes
        int w_size = 4*n_size; //number of weights
        std::vector<std::vector<double>> node_priors {n_size};
        std::vector<std::vector<double>> weight_priors {w_size};
        std::vector<double> jitter_priors {4}; //4 datasets with jitter
        std::vector<std::vector<double>> mean_priors {3}; //3 remaining means
        std::vector<std::string> mean_type {3}; //type of each mean 
        //QPkernel *kernel;
        //HODLR_Tree<QPkernel> *A;

        unsigned int staleness;

    public:
        RVmodel();
        double extra_sigma;
        void save_setup();

        /* Generate the point from the prior */
        void from_prior(DNest4::RNG& rng);

        /* Metropolis-Hastings proposals */
        double perturb(DNest4::RNG& rng);

        /* Likelihood function */
        double log_likelihood() const;

        /* Print to stream */
        void print(std::ostream& out) const;

        /* Return string with column information */
        std::string description() const;

};

#endif

