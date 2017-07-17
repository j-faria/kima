#include "MyConditionalPrior.h"
#include "DNest4.h"
#include "Utils.h"
#include <cmath>

using namespace std;
using namespace DNest4;

ModifiedJeffreys Pprior(1.0, 9999.); // days
ModifiedJeffreys Kprior(1.0, 999.); // m/s

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
// vec[3] = v0
// vec[4] = viewing angle

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
	if(vec[1] < 0. ||
	   vec[2] < 0. || vec[2] > 2.*M_PI ||
	   //vec[3] < 0. || vec[3] > 0.8189776 ||
	   vec[3] < 0. || vec[3] > 1.0 ||
	   vec[4] < 0. || vec[4] > 2.*M_PI)
		 return -1E300;

	return 0.;
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
	//cout << vec[0] << endl;
	vec[0] = Pprior.cdf_inverse(vec[0]);
	vec[1] = Kprior.cdf_inverse(vec[1]);
	vec[2] = 2.*M_PI*vec[2];
	//vec[3] = vec[3]*1.0;
	vec[4] = 2.*M_PI*vec[4];
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
	vec[0] = Pprior.cdf(vec[0]);
	vec[1] = Kprior.cdf(vec[1]);
	vec[2] = vec[2]/(2.*M_PI);
	//vec[3] = vec[3]/1.0;
	vec[4] = vec[4]/(2.*M_PI);
}

void MyConditionalPrior::print(std::ostream& out) const
{
	//out<<center<<' '<<width<<' '<<mu<<' ';
}

