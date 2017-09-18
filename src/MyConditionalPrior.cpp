#include "MyConditionalPrior.h"
#include "DNest4.h"
#include "Utils.h"
#include <cmath>

using namespace std;
using namespace DNest4;

// ModifiedJeffreys Pprior(1.0, 9999.); // days
Jeffreys Pprior(1.0, 1E4); // days
ModifiedJeffreys Kprior(1.0, 999.); // m/s
TruncatedRayleigh eprior(0.2, 0.0, 1.0);
Uniform phiprior(0.0, 2*M_PI);
Uniform wprior(0.0, 2*M_PI);


MyConditionalPrior::MyConditionalPrior()
{

}

void MyConditionalPrior::from_prior(RNG& rng)
// these are the quantile functions (inverse of the cdf) for the hyperparameters
{
	// no hyperparameters
}

double MyConditionalPrior::perturb_hyperparameters(RNG& rng)
{
	double logH = 0.;
	return logH;
}

// vec[0] = period
// vec[1] = amplitude
// vec[2] = phase
// vec[3] = ecc
// vec[4] = viewing angle

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
	if(vec[0] < 1.25 || vec[0] > 1E4 ||
	   vec[1] < 0. ||
	   vec[2] < 0. || vec[2] > 2.*M_PI ||
	   vec[3] < 0. || vec[3] >= 1.0 ||
	   vec[4] < 0. || vec[4] > 2.*M_PI)
		 return -1E300;

	return Pprior.log_pdf(vec[0]) + 
	       Kprior.log_pdf(vec[1]) + 
	       phiprior.log_pdf(vec[2]) + 
	       eprior.log_pdf(vec[3]) + 
	       wprior.log_pdf(vec[4]);
	//return 0.;
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec, int id) const
{
	//cout << id << endl;
	vec[0] = Pprior.cdf_inverse(vec[0]);
	vec[1] = Kprior.cdf_inverse(vec[1]);
	vec[2] = phiprior.cdf_inverse(vec[2]); //2.*M_PI*vec[2];
	//vec[3] = vec[3]*1.0;
	vec[3] = eprior.cdf_inverse(vec[3]);
	vec[4] = wprior.cdf_inverse(vec[4]); //2.*M_PI*vec[4];
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec, int id) const
{
	vec[0] = Pprior.cdf(vec[0]);
	vec[1] = Kprior.cdf(vec[1]);
	vec[2] = phiprior.cdf(vec[2]); //vec[2]/(2.*M_PI);
	//vec[3] = vec[3]/1.0;
	vec[3] = eprior.cdf(vec[3]);
	vec[4] = wprior.cdf(vec[4]); //vec[4]/(2.*M_PI);
}

void MyConditionalPrior::print(std::ostream& out) const
{
	//out<<center<<' '<<width<<' '<<mu<<' ';
}

