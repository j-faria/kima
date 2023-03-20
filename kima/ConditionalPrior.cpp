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

BinariesConditionalPrior::BinariesConditionalPrior()
{
    if (!Pprior)
        Pprior = make_shared<LogUniform>(1., 1e5);
    if (!Kprior)
        Kprior = make_shared<ModifiedLogUniform>(1., 1e3);

    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wprior)
        wprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wdotprior)
        wdotprior = make_shared<Gaussian>(0, 100);
}

void BinariesConditionalPrior::from_prior(RNG& rng) {}

double BinariesConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    return 0.0;
}

// vec[0] = period
// vec[1] = amplitude
// vec[2] = phase
// vec[3] = ecc
// vec[4] = viewing angle
// vec[5] = rate of change of pericentre angle

double BinariesConditionalPrior::log_pdf(const std::vector<double>& vec) const
{

    if(vec[0] < 1. || vec[0] > 1E4 ||
        vec[1] < 0. ||
        vec[2] < 0. || vec[2] > 2.*M_PI ||
        vec[3] < 0. || vec[3] >= 1.0 ||
        vec[4] < 0. || vec[4] > 2.*M_PI)
            return -1E300;

    return Pprior->log_pdf(vec[0]) + 
           Kprior->log_pdf(vec[1]) + 
           phiprior->log_pdf(vec[2]) + 
           eprior->log_pdf(vec[3]) + 
           wprior->log_pdf(vec[4]) + 
           wdotprior->log_pdf(vec[5]);
}

void BinariesConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    vec[0] = Pprior->cdf_inverse(vec[0]);
    vec[1] = Kprior->cdf_inverse(vec[1]);
    vec[2] = phiprior->cdf_inverse(vec[2]);
    vec[3] = eprior->cdf_inverse(vec[3]);
    vec[4] = wprior->cdf_inverse(vec[4]);
    vec[5] = wdotprior->cdf_inverse(vec[5]);
}

void BinariesConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    vec[0] = Pprior->cdf(vec[0]);
    vec[1] = Kprior->cdf(vec[1]);
    vec[2] = phiprior->cdf(vec[2]);
    vec[3] = eprior->cdf(vec[3]);
    vec[4] = wprior->cdf(vec[4]);
    vec[5] = wdotprior->cdf(vec[5]);
}

void BinariesConditionalPrior::print(std::ostream& out) const {}


/*****************************************************************************/

RVMixtureConditionalPrior::RVMixtureConditionalPrior()
{
    // if (!Kt_prior)
    // LN(12 √2 σRV, 2) truncated to (0, 2 km/s)
    // Kt_prior = make_shared<TruncatedLogNormal>(2*2*3*sqrt(2)*0.36/sqrt(240), 2.0, 0.0, 20);
    Kt_prior = make_shared<TruncatedGaussian>(0, 2*2*3*sqrt(2)*0.36/sqrt(240), 0.0, 20.0);
    // Kt_prior = make_shared<Fixed>(1);

    // if (!Kmax_prior)
    Kmax_prior = make_shared<LogUniform>(1.0, 20); // will be reset later
    // Kmax_prior = make_shared<Fixed>(2e3);

    if (!Pprior)
        Pprior = make_shared<LogUniform>(1.0, 1000.0);


    if (!K1prior)
        K1prior = make_shared<Uniform>(0.0, 1.0);
        //                                  Kt
    if (!K2prior)
        K2prior = make_shared<LogUniform>(1.0, 20);
        //                                Kt   Kmax

    if (!eprior)
        eprior = make_shared<Kumaraswamy>(0.867, 3.03);
        // eprior = make_shared<Uniform>(0, 1);

    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wprior)
        wprior = make_shared<Uniform>(0, 2*M_PI);

    if (!Lprior)
        Lprior = make_shared<BetaBinom>(1, 1.0, 1.0);
        // Lprior = make_shared<Uniform>(0.0, 1.0);
        // Lprior = make_shared<Beta>(1.0, 1.0);
}

void RVMixtureConditionalPrior::from_prior(RNG& rng)
{
    // sample Kt
    Kt = Kt_prior->generate(rng);
    // reset prior for Kmax
    Kmax_prior->setpars(Kt, 20);
    // sample Kmax
    Kmax = Kmax_prior->generate(rng);
    // reset priors for K1 and K2
    K1prior->setpars(0.0, Kt);
    K2prior->setpars(Kt, Kmax);
}

double RVMixtureConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    // make sure priors are correct
    // ? this is to guarantee that after a failed proposal (which updates the
    // ? priors) they are reset to the correct values
    Kmax_prior->setpars(Kt, 20);
    K1prior->setpars(0.0, Kt);
    K2prior->setpars(Kt, Kmax);

    double logH = 0.;

    // perturb Kt
    Kt_prior->perturb(Kt, rng);

    do {
        // unwrap the perturb method to update the Kmax prior in between
        Kmax = Kmax_prior->cdf(Kmax);           // perturb
        Kmax += rng.randh();                    // |
        wrap(Kmax, 0.0, 1.0);                   // |
        Kmax_prior->setpars(Kt, 20);            // reset prior
        Kmax = Kmax_prior->cdf_inverse(Kmax);   // |
    } while (Kmax <= Kt);

    // reset prior for K1 and K2
    K1prior->setpars(0.0, Kt);
    K2prior->setpars(Kt, Kmax);

    return logH;
}

double RVMixtureConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    // P, K, φ, e, w, λ = vec
    // 0  1  2  3  4  5

    // K1prior->setpars(0.0, Kt);
    // K2prior->setpars(Kt, Kmax);

    // double x1 = exp(K1prior->log_pdf(vec[1]));
    // double x2 = exp(K2prior->log_pdf(vec[1]));
    // double k = log(vec[5] * x1 + (1 - vec[5]) * x2);
    double k;
    if (vec[5] == 0.0)
        k = K2prior->log_pdf(vec[1]);
    else if (vec[5] == 1.0)
        k = K1prior->log_pdf(vec[1]);

    return Pprior->log_pdf(vec[0]) + 
           k +
           phiprior->log_pdf(vec[2]) + 
           eprior->log_pdf(vec[3]) + 
           wprior->log_pdf(vec[4]) +
           Lprior->log_pdf(vec[5]);
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
    // P, K, φ, e, w, λ = vec
    // 0  1  2  3  4  5

    // K1prior->setpars(0.0, Kt);
    // K2prior->setpars(Kt, Kmax);

    // period, use inverse of cdf
    vec[0] = Pprior->cdf_inverse(vec[0]);

    vec[5] = Lprior->cdf_inverse(vec[5]);
    // double a = bisect_mixture_cdf(vec[1], vec[5]);
    // vec[1] = a;
    if (vec[5] == 0.0)
        vec[1] = K2prior->cdf_inverse(vec[1]);
    else if (vec[5] == 1.0)
        vec[1] = K1prior->cdf_inverse(vec[1]);

    vec[2] = phiprior->cdf_inverse(vec[2]); // φ
    vec[3] = eprior->cdf_inverse(vec[3]);   // e
    vec[4] = wprior->cdf_inverse(vec[4]);   // w

}

void RVMixtureConditionalPrior::to_uniform(std::vector<double> &vec) const {
    // K1prior->setpars(0.0, Kt);
    // K2prior->setpars(Kt, Kmax);

    vec[0] = Pprior->cdf(vec[0]);

    vec[5] = Lprior->cdf(vec[5]);
    vec[1] = vec[5] * K1prior->cdf(vec[1]) + (1 - vec[5]) * K2prior->cdf(vec[1]);

    vec[2] = phiprior->cdf(vec[2]);
    vec[3] = eprior->cdf(vec[3]);
    vec[4] = wprior->cdf(vec[4]);
}

void RVMixtureConditionalPrior::print(std::ostream& out) const
{
    out << Kt << ' ' << Kmax << ' ';
}



