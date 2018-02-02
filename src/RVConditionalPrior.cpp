#include "RVConditionalPrior.h"
#include "DNest4.h"
#include "Utils.h"
#include <cmath>
#include <typeinfo>

using namespace std;
using namespace DNest4;


extern ContinuousDistribution *log_muP_prior;
extern ContinuousDistribution *wP_prior;
extern ContinuousDistribution *log_muK_prior;

extern ContinuousDistribution *Pprior;
extern ContinuousDistribution *Kprior;

extern ContinuousDistribution *eprior;
extern ContinuousDistribution *phiprior;
extern ContinuousDistribution *wprior;


RVConditionalPrior::RVConditionalPrior()
{
}


void RVConditionalPrior::from_prior(RNG& rng)
{
    if(hyperpriors)
    {
        center = log_muP_prior->generate(rng);
        width = wP_prior->generate(rng);
        muK = exp(log_muK_prior->generate(rng));
    }
}

double RVConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    double logH = 0.;

    if(hyperpriors)
    {
        int which = rng.rand_int(3);

        if(which == 0)
            log_muP_prior->perturb(center, rng);
        else if(which == 1)
            wP_prior->perturb(width, rng);
        else
        {
            muK = log(muK);
            log_muK_prior->perturb(muK, rng);
            muK = exp(muK);
        }
    }

    return logH;
}

// vec[0] = period
// vec[1] = amplitude
// vec[2] = phase
// vec[3] = ecc
// vec[4] = viewing angle

double RVConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    if(hyperpriors)
    {
        if(vec[2] < 0. || vec[2] > 2.*M_PI ||
           vec[3] < 0. || vec[3] >= 1.0 ||
           vec[4] < 0. || vec[4] > 2.*M_PI)
             return -1E300;

        Pprior->setpars(center, width);
        Kprior->setpars(muK);
    }
    else
    {
        if(vec[0] < 1. || vec[0] > 1E4 ||
           vec[1] < 0. ||
           vec[2] < 0. || vec[2] > 2.*M_PI ||
           vec[3] < 0. || vec[3] >= 1.0 ||
           vec[4] < 0. || vec[4] > 2.*M_PI)
             return -1E300;
    }

    return Pprior->log_pdf(vec[0]) + 
           Kprior->log_pdf(vec[1]) + 
           phiprior->log_pdf(vec[2]) + 
           eprior->log_pdf(vec[3]) + 
           wprior->log_pdf(vec[4]);
}

void RVConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    if(hyperpriors)
    {
        Pprior->setpars(center, width);
        Kprior->setpars(muK);
    }

    vec[0] = Pprior->cdf_inverse(vec[0]);
    vec[1] = Kprior->cdf_inverse(vec[1]);
    vec[2] = phiprior->cdf_inverse(vec[2]);
    vec[3] = eprior->cdf_inverse(vec[3]);
    vec[4] = wprior->cdf_inverse(vec[4]);
}

void RVConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    if(hyperpriors)
    {
        Pprior->setpars(center, width);
        Kprior->setpars(muK);
    }

    vec[0] = Pprior->cdf(vec[0]);
    vec[1] = Kprior->cdf(vec[1]);
    vec[2] = phiprior->cdf(vec[2]);
    vec[3] = eprior->cdf(vec[3]);
    vec[4] = wprior->cdf(vec[4]);
}

void RVConditionalPrior::print(std::ostream& out) const
{
    if(hyperpriors)
        out<<center<<' '<<width<<' '<<muK<<' ';
}
