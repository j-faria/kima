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


/*
if(hyperpriors)
{
    // Cauchy prior centered on log(365 days), scale=1
    // for log(muP), with muP in days
    // -- truncated to (~-15.1, ~26.9) --
    TruncatedCauchy log_muP_prior(log(365), 1., log(365)-21, log(365)+21);

    // Uniform prior between 0.1 and 3
    // for wP
    Uniform wP_prior(0.1, 3.);
    
    // Cauchy prior centered on log(1), scale=1
    // for log(muK), with muK in m/s
    // -- truncated to (-21, 21) --
    // NOTE: we actually sample on muK itself, just the prior is for log(muK). 
    // Confusing, I know...
    TruncatedCauchy log_muK_prior(0., 1., 0.-21, 0.+21);

    Laplace Pprior;
    Exponential Kprior;

}
else
{
*/
    //extern Jeffreys Pprior;
    extern ContinuousDistribution *Pprior;
    extern ContinuousDistribution *Kprior;
    // Jeffreys Pprior(1.0, 1E7); // days
    //ModifiedJeffreys Kprior(1.0, 1E4); // m/s
    //Uniform Kprior(0., 20.); // m/s

    extern ContinuousDistribution *eprior;
    extern ContinuousDistribution *phiprior;
    extern ContinuousDistribution *wprior;
    //TruncatedRayleigh eprior(0.2, 0.0, 1.0);
    //TruncatedNormal eprior(0, 0.3, 0., 1.);


RVConditionalPrior::RVConditionalPrior()
{
}


void RVConditionalPrior::from_prior(RNG& rng)
{
    //cout << "called RVConditionalPrior::from_prior !!!" << endl;
    if(hyperpriors)
    {
        center = log_muP_prior->rvs(rng);
        width = wP_prior->rvs(rng);
        muK = exp(log_muK_prior->rvs(rng));
    }
}

double RVConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    double logH = 0.;

    if(hyperpriors)
    {
        int which = rng.rand_int(3);

        if(which == 0)
        {
            log_muP_prior->perturb(center, rng);
        }        
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
    //cout << "type of Pprior:" << typeid(Pprior).name() << endl;
    //cout << "called RVConditionalPrior::log_pdf !!!" << endl;

    // Determine whether to use the global priors for P/K
    // or whether to use ones based on the hyperparameters
    Laplace l(center, width);
    Exponential e(muK);
    ContinuousDistribution* _Pprior = Pprior;
    ContinuousDistribution* _Kprior = Kprior;

    if(hyperpriors)
    {
        if(vec[2] < 0. || vec[2] > 2.*M_PI ||
           vec[3] < 0. || vec[3] >= 1.0 ||
           vec[4] < 0. || vec[4] > 2.*M_PI)
             return -1E300;

        _Pprior = &l;
        _Kprior = &e;
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

    return _Pprior->log_pdf(vec[0]) + 
           _Kprior->log_pdf(vec[1]) + 
           phiprior->log_pdf(vec[2]) + 
           eprior->log_pdf(vec[3]) + 
           wprior->log_pdf(vec[4]);
}

void RVConditionalPrior::from_uniform(std::vector<double>& vec, int id) const
{
    // Determine whether to use the global priors for P/K
    // or whether to use ones based on the hyperparameters
    Laplace l(center, width);
    Exponential e(muK);
    ContinuousDistribution* _Pprior = Pprior;
    ContinuousDistribution* _Kprior = Kprior;
    if(hyperpriors)
    {
        _Pprior = &l;
        _Kprior = &e;
    }

    vec[0] = _Pprior->cdf_inverse(vec[0]);
    vec[1] = _Kprior->cdf_inverse(vec[1]);
    vec[2] = phiprior->cdf_inverse(vec[2]); //2.*M_PI*vec[2];
    vec[3] = eprior->cdf_inverse(vec[3]);
    vec[4] = wprior->cdf_inverse(vec[4]); //2.*M_PI*vec[4];
}

void RVConditionalPrior::to_uniform(std::vector<double>& vec, int id) const
{
    // Determine whether to use the global priors for P/K
    // or whether to use ones based on the hyperparameters
    Laplace l(center, width);
    Exponential e(muK);
    ContinuousDistribution* _Pprior = Pprior;
    ContinuousDistribution* _Kprior = Kprior;
    if(hyperpriors)
    {
        _Pprior = &l;
        _Kprior = &e;
    }

    vec[0] = _Pprior->cdf(vec[0]);
    vec[1] = _Kprior->cdf(vec[1]);
    vec[2] = phiprior->cdf(vec[2]); //vec[2]/(2.*M_PI);
    vec[3] = eprior->cdf(vec[3]);
    vec[4] = wprior->cdf(vec[4]); //vec[4]/(2.*M_PI);
}

void RVConditionalPrior::print(std::ostream& out) const
{
    if(hyperpriors)
        out<<center<<' '<<width<<' '<<muK<<' ';
}


void RVConditionalPrior::print0(std::ostream& out) const
{
    out<<0.<<' '<<0.<<' '<<0.<<' ';
}
