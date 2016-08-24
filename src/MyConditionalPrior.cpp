#include "MyConditionalPrior.h"
#include "Utils.h"
#include <cmath>

using namespace DNest4;

MyConditionalPrior::MyConditionalPrior()
{

}

void MyConditionalPrior::from_prior(RNG& rng)
// these are the quantile functions (inverse of the cdf)
{
	// Cauchy prior centered on 5.901 = log(365 days), scale=1
	// truncated to (-15.3, 27.1)
	center = 5.901 + tan(M_PI*(0.97*rng.rand() - 0.485));
	// uniform prior between 0.1 and 3
	width = 0.1 + 2.9*rng.rand();
	// standard Cauchy prior for log(mu)
	// truncated to (-21.2, 21.2)
	/*mu = exp(tan(M_PI*(0.97*rng.rand() - 0.485)));*/
	// or Cauchy prior centered on mean error bar, scale=1 for log(mu)
	// truncated to (-19.2, 23.2)
	/*mu = exp(2. + tan(M_PI*(0.97*rand() - 0.485)));*/
	// or Cauchy prior centered on -6.908 = log(0.001km/s) for log(mu) -- if mu is in km/s
	// truncated to (-28.1, 14.3)
	mu = exp(-6.908 + tan(M_PI*(0.97*rand() - 0.485)));

}

double MyConditionalPrior::perturb_hyperparameters(RNG& rng)
{
	double logH = 0.;

	int which = rng.rand_int(3);

	if(which == 0)
	{
		center = (atan(center - 5.901)/M_PI + 0.485)/0.97; // cdf
		center += rng.randh();
		wrap(center, 0., 1.);
		center = 5.901 + tan(M_PI*(0.97*center - 0.485)); // quantile
	}
	else if(which == 1)
	{
		width += 2.9*rng.randh();
		wrap(width, 0.1, 3.);
	}
	else
	{
/*		mu = log(mu);
		mu = (atan(mu)/M_PI + 0.485)/0.97; // cdf
		mu += rng.randh();
		wrap(mu, 0., 1.);
		mu = tan(M_PI*(0.97*mu - 0.485)); // quantile
		mu = exp(mu);
*/

/*		mu = log(mu);
		mu = (atan(mu - 2.)/M_PI + 0.485)/0.97; // cdf
		mu += randh();
		wrap(mu, 0., 1.);
		mu = 2. + tan(M_PI*(0.97*mu - 0.485)); // quantile
		mu = exp(mu);
*/

		mu = log(mu);
		mu = (atan(mu + 6.908)/M_PI + 0.485)/0.97; // cdf
		mu += rng.randh();
		wrap(mu, 0., 1.);
		mu = -6.908 + tan(M_PI*(0.97*mu - 0.485)); // quantile
		mu = exp(mu);
	}
	return logH;
}

// vec[0] = "position" (log-period)
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

	return  -log(2.*width) - abs(vec[0] - center)/width
		-log(mu) - vec[1]/mu;
		//+ 2.1*log(1. - vec[3]/0.995);
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
	if(vec[0] < 0.5)
		vec[0] = center + width*log(2.*vec[0]);
	else
		vec[0] = center - width*log(2. - 2.*vec[0]);
	vec[1] = -mu*log(1. - vec[1]);
	vec[2] = 2.*M_PI*vec[2];
	//vec[3] = 1. - pow(1. - 0.995*vec[3], 1./3.1);
	vec[4] = 2.*M_PI*vec[4];
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
	if(vec[0] < center)
		vec[0] = 0.5*exp((vec[0] - center)/width);
	else
		vec[0] = 1. - 0.5*exp((center - vec[0])/width);
	vec[1] = 1. - exp(-vec[1]/mu);
	vec[2] = vec[2]/(2.*M_PI);
	//vec[3] = 1. - pow(1. - vec[3]/0.995, 3.1);
	vec[4] = vec[4]/(2.*M_PI);
}

void MyConditionalPrior::print(std::ostream& out) const
{
	out<<center<<' '<<width<<' '<<mu<<' ';
}

