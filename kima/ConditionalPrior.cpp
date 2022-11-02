#include "ConditionalPrior.h"

using namespace std;
using namespace DNest4;

RVConditionalPrior::RVConditionalPrior():hyperpriors(false)
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
            Pprior = make_shared<LogUniform>(1.0, 1e3);
        if (!Kprior)
            Kprior = make_shared<ModifiedLogUniform>(1., 1e3);
    }

    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wprior)
        wprior = make_shared<Uniform>(0, 2*M_PI);
}

void RVConditionalPrior::use_hyperpriors()
{
    hyperpriors = true;
    if (!log_muP_prior)
        /// By default, Cauchy prior centered on log(365 days), scale=1
        /// for log(muP), with muP in days, truncated to (~-15.1, ~26.9)
        log_muP_prior = make_shared<TruncatedCauchy>(log(365), 1., log(365)-21, log(365)+21);

    /// By default, uniform prior for wP
    if (!wP_prior)
        wP_prior = make_shared<Uniform>(0.1, 3);

    /// By default, Cauchy prior centered on log(1), scale=1
    /// for log(muK), with muK in m/s, truncated to (-21, 21)
    /// NOTE: we actually sample on muK itself, just the prior is for log(muK)
    if (!log_muK_prior)
        log_muK_prior = make_shared<TruncatedCauchy>(0., 1., 0.-21, 0.+21);

    Pprior = make_shared<Laplace>();
    Kprior = make_shared<Exponential>();
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

double RVConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    if(hyperpriors)
    {
        Pprior->setpars(center, width);
        Kprior->setpars(muK);
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


/*****************************************************************************/

RVMixtureConditionalPrior::RVMixtureConditionalPrior()
{
    if (!tau1_prior)
        tau1_prior = make_shared<Uniform>(0.0, 1.0);
    if (!tau2_prior)
        tau2_prior = make_shared<Uniform>(0.0, 20.0);

    if (!Pprior)
        Pprior = make_shared<LogUniform>(1.0, 1000.0);


    if (!K1prior)
        K1prior = make_shared<TruncatedGaussian>(0.0, 1.0, 0.0, 1.0 / 0.0);
        //                                            tau1
    if (!K2prior)
        K2prior = make_shared<TruncatedGaussian>(0.0, 1.0, 0.0, 1.0 / 0.0);
        //                                            tau2

    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wprior)
        wprior = make_shared<Uniform>(0, 2*M_PI);

    if (!Lprior)
        Lprior = make_shared<Kumaraswamy>(0.5, 0.5);
}

void RVMixtureConditionalPrior::from_prior(RNG& rng)
{
    tau1 = tau1_prior->generate(rng);
    tau2 = tau2_prior->generate(rng);
    K1prior->setpars(0.0, tau1);
    K2prior->setpars(0.0, tau2);
}

double RVMixtureConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    double logH = 0.;

    int which = rng.rand_int(2);

    if (which == 0)
        tau1_prior->perturb(tau1, rng);
    else if (which == 1)
        tau2_prior->perturb(tau2, rng);

    return logH;
}

double RVMixtureConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    K1prior->setpars(0.0, tau1);
    K2prior->setpars(0.0, tau2);

    double x1 = exp(K1prior->log_pdf(vec[1]));
    double x2 = exp(K2prior->log_pdf(vec[1]));
    double k = log(vec[5] * x1 + (1 - vec[5]) * x2);

    return Pprior->log_pdf(vec[0]) + 
           k +
           phiprior->log_pdf(vec[2]) + 
           eprior->log_pdf(vec[3]) + 
           wprior->log_pdf(vec[4]);
}

double RVMixtureConditionalPrior::bisect_mixture_cdf(double p, double L) const
{
    double a = K1prior->cdf_inverse(p);
    double b = K2prior->cdf_inverse(p);
    double c = a;
    while ((b - a) >= 1e-5)
    {
        // Find middle point
        c = 0.5 * (a + b);
        double cdf_c = L * K1prior->cdf(c) + (1 - L) * K2prior->cdf(c);
        double cdf_a = L * K1prior->cdf(a) + (1 - L) * K2prior->cdf(a);
        if (cdf_c - p == 0.0)
            break;
        else if ((cdf_c - p) * (cdf_a - p) < 0.0)
            b = c;
        else
            a = c;
    }
    return c;
}

void RVMixtureConditionalPrior::from_uniform(std::vector<double>& vec) const {

// void RVMixtureConditionalPrior::from_uniform(DNest4::RNG &rng, std::vector<double>& vec) const
// {
    K1prior->setpars(0.0, tau1);
    K2prior->setpars(0.0, tau2);

    vec[0] = Pprior->cdf_inverse(vec[0]);

    vec[5] = Lprior->cdf_inverse(vec[5]);
    double a = bisect_mixture_cdf(vec[1], vec[5]);
    vec[1] = a;

    vec[2] = phiprior->cdf_inverse(vec[2]);
    vec[3] = eprior->cdf_inverse(vec[3]);
    vec[4] = wprior->cdf_inverse(vec[4]);

}

void RVMixtureConditionalPrior::to_uniform(std::vector<double> &vec) const {

    K1prior->setpars(0.0, tau1);
    K2prior->setpars(0.0, tau2);

    vec[0] = Pprior->cdf(vec[0]);

    vec[5] = Lprior->cdf(vec[5]);
    vec[1] = vec[5] * K1prior->cdf(vec[1]) + (1 - vec[5]) * K2prior->cdf(vec[1]);

    vec[2] = phiprior->cdf(vec[2]);
    vec[3] = eprior->cdf(vec[3]);
    vec[4] = wprior->cdf(vec[4]);
}

void RVMixtureConditionalPrior::print(std::ostream& out) const
{
    out << tau1 << ' ' << tau2 << ' ';
}



