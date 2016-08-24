#ifndef DNest4_MyModel
#define DNest4_MyModel

#include <vector>
#include "MyConditionalPrior.h"
#include "RJObject/RJObject.h"
#include "RNG.h"
#include "Data.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "HODLR_Matrix.hpp"
#include "HODLR_Tree.hpp"

class QPkernel : public HODLR_Matrix {
	/* This implements the quasi-periodic kernel */
	friend class MyModel;

    public:
        QPkernel (vector<double> time) 
            : time_(time) {};

        void set_hyperpars (double eta1, double eta2, double eta3, double eta4)
            {
            	eta1_ = eta1;
            	eta2_ = eta2;
            	eta3_ = eta3;
            	eta4_ = eta4;
            };

        double get_Matrix_Entry(const unsigned i, const unsigned j) {
            double d = time_[i] - time_[j];
            double Cij = eta1_*eta1_*exp(-0.5*pow(d/eta2_, 2) 
                                        -2.0*pow(sin(M_PI*d/eta3_)/eta4_, 2) );
            
            return Cij;
            //return theta_[0] * exp(-0.5 * d * d / (theta_[0]));
        }

    private:
    	double eta1_, eta2_, eta3_, eta4_;
        vector<double> time_;
};



class MyModel
{
	private:
		DNest4::RJObject<MyConditionalPrior> objects;

		double background;
		std::vector<double> offsets;

		double extra_sigma;

		// Parameters for the quasi-periodic extra noise
		double eta1, eta2, eta3, eta4, eta5;

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

		QPkernel *kernel;
		HODLR_Tree<QPkernel> *A;

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

