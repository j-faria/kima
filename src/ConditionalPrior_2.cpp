// (c) 2019 João Faria
// This file is part of kima, which is licensed under the MIT license (see LICENSE for details)

#include "ConditionalPrior_2.h"
#include "DNest4.h"
#include "Utils.h"
#include <cmath>
#include <typeinfo>

using namespace std;
using namespace DNest4;

RVConditionalPrior_2::RVConditionalPrior_2()
{
    if (hyperpriors){
        if (!log_muP_prior)
            /** 
             * By default, Cauchy prior centered on log(365 days), scale=1
             * for log(muP), with muP in days
             * truncated to (~-15.1, ~26.9)
            */
            log_muP_prior = make_shared<TruncatedCauchy>(log(365), 1., log(365)-21, log(365)+21);
        // TruncatedCauchy *log_muP_prior = new TruncatedCauchy(log(365), 1., log(365)-21, log(365)+21);

        /**
         * By default, uniform prior for wP
        */
        if (!wP_prior)
            wP_prior = make_shared<Uniform>(0.1, 3);
        // Uniform *wP_prior = new Uniform(0.1, 3.);
    

        /**
         * By default, Cauchy prior centered on log(1), scale=1
         * for log(muK), with muK in m/s
         * truncated to (-21, 21)
         * NOTE: we actually sample on muK itself, just the prior is for log(muK)
        */
        if (!log_muK_prior)
            log_muK_prior = make_shared<TruncatedCauchy>(0., 1., 0.-21, 0.+21);
        // TruncatedCauchy *log_muK_prior = new TruncatedCauchy(0., 1., 0.-21, 0.+21);

        Pprior = make_shared<Laplace>();
        Kprior = make_shared<Exponential>();
    }
    else {
        if (!Pprior)
            Pprior = make_shared<LogUniform>(1., 1e5);
        if (!Kprior)
            Kprior = make_shared<ModifiedLogUniform>(1., 1e3);
    }


    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wprior)
        wprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wdotprior)
        wdotprior = make_shared<Gaussian>(0, 100);
}


void RVConditionalPrior_2::from_prior(RNG& rng)
{
    if(hyperpriors)
    {
        center = log_muP_prior->generate(rng);
        width = wP_prior->generate(rng);
        muK = exp(log_muK_prior->generate(rng));
    }
}

double RVConditionalPrior_2::perturb_hyperparameters(RNG& rng)
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
// vec[5] = rate of change of pericentre angle

double RVConditionalPrior_2::log_pdf(const std::vector<double>& vec) const
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
           wprior->log_pdf(vec[4]) + 
           wdotprior->log_pdf(vec[5]);
}

void RVConditionalPrior_2::from_uniform(std::vector<double>& vec) const
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
    vec[5] = wdotprior->cdf_inverse(vec[5]);
}

void RVConditionalPrior_2::to_uniform(std::vector<double>& vec) const
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
    vec[5] = wdotprior->cdf(vec[5]);
}

void RVConditionalPrior_2::print(std::ostream& out) const
{
    if(hyperpriors)
        out<<center<<' '<<width<<' '<<muK<<' ';
}
