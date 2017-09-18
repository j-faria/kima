#ifndef DNest4_MyModel
#define DNest4_MyModel

#include <vector>
#include "MyConditionalPrior.h"
#include "RJObject/RJObject.h"
#include "RNG.h"
#include "Data.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "celerite/celerite.h"


class MyModel
{
    private:
        DNest4::RJObject<MyConditionalPrior> objects;

        double background;
        //std::vector<double> offsets;
        double slope, quad;

        double extra_sigma;

        // Parameters for the quasi-periodic extra noise
        double eta1, eta2, eta3, eta4, eta5;
        double a,b,c,P;

        celerite::solver::CholeskySolver<double> solver;
        Eigen::VectorXd alpha_real,
                 beta_real,
                 alpha_complex_real,
                 alpha_complex_imag,
                 beta_complex_real,
                 beta_complex_imag;

        // The signal
        std::vector<long double> mu;
        void calculate_mu();

        // eccentric and true anomalies
        double ecc_anomaly(double time, double prd, double ecc, double peri_pass);
        double eps3(double e, double M, double x);
        double keplerstart3(double e, double M);
        double true_anomaly(double time, double prd, double ecc, double peri_pass);

        // The covariance matrix for the data
        Eigen::MatrixXd C;
        void calculate_C();

        //QPkernel *kernel;
        //HODLR_Tree<QPkernel> *A;

        unsigned int staleness;

    public:
        MyModel();

        void setupHODLR();

        // Generate the point from the prior
        void from_prior(DNest4::RNG& rng);

        // Metropolis-Hastings proposals
        double perturb(DNest4::RNG& rng);

        // Likelihood function
        double log_likelihood() const;

        // Print to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;
};

#endif

