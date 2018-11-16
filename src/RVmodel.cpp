#include "RVmodel.h"
#include "RVConditionalPrior.h"
#include "DNest4.h"
#include "RNG.h"
#include "Utils.h"
#include "Data.h"
#include <cmath>
#include <limits>
#include <fstream>
#include <chrono>
#include <time.h> 
#include "GPRN.h"

using namespace std;
using namespace Eigen;
using namespace DNest4;

#define TIMING false

extern ContinuousDistribution *Cprior; // systematic velocity, m/s
extern ContinuousDistribution *Jprior; // additional white noise, m/s
extern ContinuousDistribution *slope_prior; // m/s/day

/* GP priors */
extern ContinuousDistribution *log_eta1_prior;
extern ContinuousDistribution *log_eta2_prior;
extern ContinuousDistribution *eta3_prior;
extern ContinuousDistribution *log_eta4_prior;

/* GPRN priors */
extern ContinuousDistribution *constant_weight;
extern ContinuousDistribution *constant_prior;
extern ContinuousDistribution *se_weight;
extern ContinuousDistribution *se_ell;
extern ContinuousDistribution *per_weight;
extern ContinuousDistribution *per_ell;
extern ContinuousDistribution *per_period;
extern ContinuousDistribution *quasi_weight;
extern ContinuousDistribution *quasi_elle;
extern ContinuousDistribution *quasi_period;
extern ContinuousDistribution *quasi_ellp;
extern ContinuousDistribution *ratq_weight;
extern ContinuousDistribution *ratq_alpha;
extern ContinuousDistribution *ratq_ell;
extern ContinuousDistribution *cos_weight;
extern ContinuousDistribution *cos_period;
extern ContinuousDistribution *exp_weight;
extern ContinuousDistribution *exp_ell;
extern ContinuousDistribution *m32_weight;
extern ContinuousDistribution *m32_ell;
extern ContinuousDistribution *m52_weight;
extern ContinuousDistribution *m52_ell;

/* from the offsets determined by Lo Curto et al. 2015 (only FGK stars)
mean, std = 14.641789473684208, 2.7783035258938971 */
Gaussian *fiber_offset_prior = new Gaussian(15., 3.);
//Uniform *fiber_offset_prior = new Uniform(0., 50.);  // old 

const double halflog2pi = 0.5*log(2.*M_PI);


void RVmodel::from_prior(RNG& rng)
{
    planets.from_prior(rng);
    planets.consolidate_diff();
    
    background = Cprior->generate(rng);
    extra_sigma = Jprior->generate(rng);

    if(obs_after_HARPS_fibers)
        fiber_offset = fiber_offset_prior->generate(rng);

    if(trend)
        slope = slope_prior->generate(rng);

    if(GP)
    {
        if(RN)
        {
            /* Generate priors accordingly to the kernels in use */
            n_size = GPRN::get_instance().node.size(); //number of nodes
            w_size = 4 * n_size; //number of weights
            /* dealing with the nodes */
            std::vector<double> priors;
            for(int i=0; i<n_size; i++)
            {
                if(GPRN::get_instance().node[i] == "C")
                {
                    prior1 = constant_prior->generate(rng);
                    node_priors[i] = {prior1};
                }
                if(GPRN::get_instance().node[i] == "SE")
                {
                    prior1 = se_ell->generate(rng);
                    node_priors[i] = {prior1};
                }
                if(GPRN::get_instance().node[i] == "P")
                {
                    prior1 = per_ell->generate(rng);
                    prior2 = per_period->generate(rng);
                    node_priors[i] = {prior1, prior2};
                }
                if(GPRN::get_instance().node[i] == "QP")
                {   
                    prior1 = quasi_elle->generate(rng);
                    prior2 = quasi_period->generate(rng);
                    prior3 = quasi_ellp->generate(rng);
                    node_priors[i] = {prior1, prior2, prior3};
                }
                if(GPRN::get_instance().node[i] == "RQ")
                {
                    prior1 = ratq_alpha->generate(rng);
                    prior2 = ratq_ell->generate(rng);
                    node_priors[i] = {prior1, prior2};
                }
                if(GPRN::get_instance().node[i] == "COS")
                {
                    prior1 = cos_period->generate(rng);
                    node_priors[i] = {prior1};
                }
                if(GPRN::get_instance().node[i] == "EXP")
                {
                    prior1 = exp_ell->generate(rng);
                    node_priors[i] = {prior1};
                }
                if(GPRN::get_instance().node[i] == "M32")
                {
                    prior1 = m32_ell->generate(rng);
                    node_priors[i] = {prior1};
                }
                if(GPRN::get_instance().node[i] == "M52")
                {
                    prior1 = m52_ell->generate(rng);
                    node_priors[i] = {prior1};
                }
            }
            /* dealing with the weights */
            if(GPRN::get_instance().weight[0] == "C")
            {
                prior2 = constant_prior->generate(rng); //all weights have the same parameters
                for(int j=0; j<w_size; j++)
                {
                    prior1 = constant_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2};
                }
            }
            if(GPRN::get_instance().weight[0] == "SE")
            {
                prior2 = se_ell->generate(rng);
                for(int j=0; j<w_size; j++)
                {
                    prior1 = se_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2};
                }
            }
            if(GPRN::get_instance().weight[0] == "P")
            {
                prior2 = per_ell->generate(rng);
                prior3 = per_period->generate(rng);
                for(int j=0; j<w_size; j++)
                {
                    prior1 = per_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2, prior3};
                }
            }
            if(GPRN::get_instance().weight[0] == "QP")
            {
                prior2 = quasi_elle->generate(rng);
                prior3 = quasi_period->generate(rng);
                prior4 = quasi_ellp->generate(rng);
                for(int j=0; j<w_size; j++)
                {
                    prior1 = quasi_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2, prior3, prior4};
                }
             }
            if(GPRN::get_instance().weight[0] == "RQ")
            {
                prior2 = ratq_alpha->generate(rng);
                prior3 = ratq_ell->generate(rng);
                for(int j=0; j<w_size; j++)
                {
                    prior1 = ratq_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2, prior3};
                }
            }
            if(GPRN::get_instance().weight[0] == "COS")
            {
                prior2 = cos_period->generate(rng);
                for(int j=0; j<w_size; j++)
                {
                    prior1 = cos_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2};
                }
            }
            if(GPRN::get_instance().weight[0] == "EXP")
            {
                prior2 = exp_ell->generate(rng);
                for(int j=0; j<w_size; j++)
                {
                    prior1 = exp_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2};
                }
            }
            if(GPRN::get_instance().weight[0] == "M32")
            {
                prior2 = m32_ell->generate(rng);
                for(int j=0; j<w_size; j++)
                {
                    prior1 = m32_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2};
                }
            }
            if(GPRN::get_instance().weight[0] == "M52")
            {
                prior2 = m52_ell->generate(rng);
                for(int j=0; j<w_size; j++)
                {
                    prior1 = m52_weight->generate(rng);
                    weight_priors[j] = {prior1, prior2};
                }
            }
        }
        else
        {
            eta1 = exp(log_eta1_prior->generate(rng));  // m/s
            eta2 = exp(log_eta2_prior->generate(rng));  // days
            eta3 = eta3_prior->generate(rng);           // days
            eta4 = exp(log_eta4_prior->generate(rng));
        }
    }
    calculate_mu();
    if(GP) calculate_C();
}


void RVmodel::calculate_C()
{
    /* if we want a GPRN the RN is set to true (RN=true) */
    if(RN)
    {
        /* data */
        auto data = Data::get_instance();
        const vector<double>& t = data.get_t();
        const vector<double>& sig = data.get_sig();
        int N = data.get_t().size();
        
        Cs = GPRN::get_instance().matrixCalculation(node_priors, weight_priors, extra_sigma);
        C = Cs[0]*0;    //I dont know why but it doesnt run if we dont define C
        
    }
    /* otherwise we just a GP and the RN is set to false (RN=false) */
    else
    {
        //data
        auto data = Data::get_instance();
        const vector<double>& t = data.get_t();
        const vector<double>& sig = data.get_sig();
        int N = data.get_t().size();
        
        for(std::size_t i=0; i<N; i++)
        {
            for(std::size_t j=i; j<N; j++)
            {
                C(i, j) = eta1*eta1*exp(-0.5*pow((t[i] - t[j])/eta2, 2) 
                            -2.0*pow(sin(M_PI*(t[i] - t[j])/eta3)/eta4, 2) );
                if(i==j)
                    {
                    C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
                    }
                else
                    C(j, i) = C(i, j);
            }
        }
    }
}


void RVmodel::calculate_mu()
{
    auto data = Data::get_instance();
    /* Get the times from the data */
    const vector<double>& t = data.get_t();

    /* Update or from scratch? */
    bool update = (planets.get_added().size() < planets.get_components().size()) &&
            (staleness <= 10);

    /* Get the components */
    const vector< vector<double> >& components = (update)?(planets.get_added()):
                (planets.get_components());
    /* at this point, components has:
        if updating: only the added planets' parameters
        if from scratch: all the planets' parameters */

    /* Zero the signal */
    if(!update) // not updating, means recalculate everything
    {
        mu.assign(mu.size(), background);
        staleness = 0;
        if(trend) 
        {
            for(size_t i=0; i<t.size(); i++)
            {
                mu[i] += slope*(t[i] - data.get_t_middle());
            }
        }

        if(obs_after_HARPS_fibers)
        {
            for(size_t i=data.index_fibers; i<t.size(); i++)
            {
                mu[i] += fiber_offset;
            }
        }


    }
    else // just updating (adding) planets
        staleness++;

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    double P, K, phi, ecc, omega, f, v, ti;
    for(size_t j=0; j<components.size(); j++)
    {
        if(hyperpriors)
            P = exp(components[j][0]);
        else
            P = components[j][0];
        
        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];

        for(size_t i=0; i<t.size(); i++)
        {
            ti = t[i];
            f = true_anomaly(ti, P, ecc, t[0]-(P*phi)/(2.*M_PI));
            v = K*(cos(f+omega) + ecc*cos(omega));
            mu[i] += v;
        }
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif    

}

double RVmodel::perturb(RNG& rng)
{
    auto data = Data::get_instance();
    const vector<double>& t = data.get_t();
    double logH = 0.;

    if(GP)
    {
        /* using a GPRN; GP=true and RN=true */
        if(RN)
        {
            if(rng.rand() <= 0.5)
            {
                logH += planets.perturb(rng);
                planets.consolidate_diff();
                calculate_mu();
            }
            else
            {
                if(rng.rand() > 0.5)
                {
                   /* dealing with the nodes */
                    std::vector<double> priors;
                    for(int i=0; i<n_size; i++)
                    {
                        if(GPRN::get_instance().node[i] == "C")
                        {
                            constant_prior->perturb(prior1, rng);
                            node_priors[i] = {prior1};
                        }
                        if(GPRN::get_instance().node[i] == "SE")
                        {
                            se_ell->perturb(prior1,rng);
                            node_priors[i] = {prior1};
                        }
                        if(GPRN::get_instance().node[i] == "P")
                        {
                            per_ell->perturb(prior1, rng);
                            per_period->perturb(prior2, rng);
                            node_priors[i] = {prior1, prior2};
                        }
                        if(GPRN::get_instance().node[i] == "QP")
                        {   
                            quasi_elle->perturb(prior1, rng);
                            quasi_period->perturb(prior2, rng);
                            quasi_ellp->perturb(prior3, rng);
                            node_priors[i] = {prior1, prior2, prior3};
                        }
                        if(GPRN::get_instance().node[i] == "RQ")
                        {
                            ratq_alpha->perturb(prior1, rng);
                            ratq_ell->perturb(prior2, rng);
                            node_priors[i] = {prior1, prior2};
                        }
                       if(GPRN::get_instance().node[i] == "COS")
                        {
                            cos_period->perturb(prior1, rng);
                            node_priors[i] = {prior1};
                        }
                      if(GPRN::get_instance().node[i] == "EXP")
                        {
                            exp_ell->perturb(prior1, rng);
                            node_priors[i] = {prior1};
                        }
                        if(GPRN::get_instance().node[i] == "M32")
                        {
                            m32_ell->perturb(prior1, rng);
                            node_priors[i] = {prior1};
                        }
                      if(GPRN::get_instance().node[i] == "M52")
                        {
                            m52_ell->perturb(prior1, rng);
                            node_priors[i] = {prior1};
                        }
                    }
                }
                if(rng.rand() < 0.5)
                {
                    /* dealing with the weights */
                    if(GPRN::get_instance().weight[0] == "C")
                    {
                        constant_prior->perturb(prior2, rng); //all weights have the same parameters
                        for(int j=0; j<w_size; j++)
                        {
                            constant_weight->perturb(prior1, rng);
                            weight_priors[j] = {prior1, prior2};
                        }
                    }
                    if(GPRN::get_instance().weight[0] == "SE")
                    {
                        se_ell->perturb(prior2, rng);
                        for(int j=0; j<w_size; j++)
                        {
                            se_weight->perturb(prior1, rng);
                            weight_priors[j] = {prior1, prior2};
                        }
                    }
                    if(GPRN::get_instance().weight[0] == "P")
                    {
                        per_ell->perturb(prior2, rng);
                        per_period->perturb(prior3, rng);
                        for(int j=0; j<w_size; j++)
                        {
                            per_weight->perturb(prior1, rng);
                            //weight_priors[j] = {prior1, prior2, prior3};
                        }
                    }
                    if(GPRN::get_instance().weight[0] == "QP")
                    {
                        quasi_elle->perturb(prior2, rng);
                        quasi_period->perturb(prior3, rng);
                        quasi_ellp->perturb(prior4, rng);
                        for(int j=0; j<w_size; j++)
                        {
                            quasi_weight->perturb(prior1, rng);
                            weight_priors[j] = {prior1, prior2, prior3, prior4};
                        }
                    }
                    if(GPRN::get_instance().weight[0] == "RQ")
                    {
                        ratq_alpha->perturb(prior2, rng);
                        ratq_ell->perturb(prior3, rng);
                        for(int j=0; j<w_size; j++)
                        {
                            ratq_weight->perturb(prior1, rng);
                            weight_priors[j] = {prior1, prior2, prior3};
                        }
                    }
                    if(GPRN::get_instance().weight[0] == "COS")
                    {
                        cos_period->perturb(prior2, rng);
                        for(int j=0; j<w_size; j++)
                        {
                            cos_weight->perturb(prior1, rng);
                            weight_priors[j] = {prior1, prior2};
                        }
                    }
                    if(GPRN::get_instance().weight[0] == "EXP")
                    {
                        exp_ell->perturb(prior2, rng);
                        for(int j=0; j<w_size; j++)
                        {
                            exp_weight->perturb(prior1, rng);
                            weight_priors[j] = {prior1, prior2};
                        }
                    }
                    if(GPRN::get_instance().weight[0] == "M32")
                    {
                        m32_ell->perturb(prior2, rng);
                        for(int j=0; j<w_size; j++)
                        {
                            m32_weight->perturb(prior1, rng);
                            weight_priors[j] = {prior1, prior2};
                        }
                    }
                    if(GPRN::get_instance().weight[0] == "M52")
                    {
                        m52_ell->perturb(prior2, rng);
                        for(int j=0; j<w_size; j++)
                        {
                            m52_weight->perturb(prior1, rng);
                            weight_priors[j] = {prior1, prior2};
                        }
                    }
                }
                calculate_C();
            }
            
            if(rng.rand() <= 0.5)
            {
                Jprior->perturb(extra_sigma, rng);
                calculate_C();
            }
            else
            {
                for(size_t i=0; i<mu.size(); i++)
                {
                    mu[i] -= background;
                    if(trend) {
                        mu[i] -= slope*(t[i]-data.get_t_middle());
                    }
                    if (obs_after_HARPS_fibers) {
                        if (i >= data.index_fibers) mu[i] -= fiber_offset;
                    }
                }

                Cprior->perturb(background, rng);

                // propose new fiber offset
                if (obs_after_HARPS_fibers) {
                    fiber_offset_prior->perturb(fiber_offset, rng);
                }

                // propose new slope
                if(trend) {
                    slope_prior->perturb(slope, rng);
                }

                for(size_t i=0; i<mu.size(); i++)
                {
                    mu[i] += background;
                    if(trend) {
                        mu[i] += slope*(t[i]-data.get_t_middle());
                    }

                    if (obs_after_HARPS_fibers) {
                        if (i >= data.index_fibers) mu[i] += fiber_offset;
                    }
                }
            }
        }
        /* using a simple GP; GP=true and RN=false */
        else
        {
            if(rng.rand() <= 0.5)
            {
                logH += planets.perturb(rng);
                planets.consolidate_diff();
                calculate_mu();
            }
            else if(rng.rand() <= 0.5)
            {
                if(rng.rand() <= 0.25)
                {
                    log_eta1 = log(eta1);
                    log_eta1_prior->perturb(log_eta1, rng);
                    eta1 = exp(log_eta1);
                }
             else if(rng.rand() <= 0.33330)
                {
                    log_eta2 = log(eta2);
                    log_eta2_prior->perturb(log_eta2, rng);
                    eta2 = exp(log_eta2);
                }
                else if(rng.rand() <= 0.5)
                {
                    eta3_prior->perturb(eta3, rng);
                }
                else
                {
                    log_eta4 = log(eta4);
                    log_eta4_prior->perturb(log_eta4, rng);
                    eta4 = exp(log_eta4);
                }
                calculate_C();
            
            }
            else if(rng.rand() <= 0.5)
            {
                Jprior->perturb(extra_sigma, rng);
                calculate_C();
            }
            else
            {
                for(size_t i=0; i<mu.size(); i++)
                {
                    mu[i] -= background;
                    if(trend) {
                        mu[i] -= slope*(t[i]-data.get_t_middle());
                    }
                    if (obs_after_HARPS_fibers) {
                        if (i >= data.index_fibers) mu[i] -= fiber_offset;
                    }
                }
                Cprior->perturb(background, rng);
                // propose new fiber offset
                if (obs_after_HARPS_fibers) {
                    fiber_offset_prior->perturb(fiber_offset, rng);
                }
                // propose new slope
                if(trend)
                {
                    slope_prior->perturb(slope, rng);
                }
                for(size_t i=0; i<mu.size(); i++)
                {
                    mu[i] += background;
                    if(trend)
                    {
                        mu[i] += slope*(t[i]-data.get_t_middle());
                    }
                    if (obs_after_HARPS_fibers)
                    {
                        if (i >= data.index_fibers) mu[i] += fiber_offset;
                    }
                }
            }
        }

    }
    else
    {
        if(rng.rand() <= 0.75)
        {
            logH += planets.perturb(rng);
            planets.consolidate_diff();
            calculate_mu();
        }
        else if(rng.rand() <= 0.5)
        {
            Jprior->perturb(extra_sigma, rng);
        }
        else
        {
            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] -= background;
                if(trend)
                {
                    mu[i] -= slope*(t[i]-data.get_t_middle());
                }
                if (obs_after_HARPS_fibers)
                {
                    if (i >= data.index_fibers) mu[i] -= fiber_offset;
                }
            }
            Cprior->perturb(background, rng);
            /* propose new fiber offset */
            if (obs_after_HARPS_fibers)
            {
                fiber_offset_prior->perturb(fiber_offset, rng);
            }
            /* propose new slope */
            if(trend)
            {
                slope_prior->perturb(slope, rng);
            }
            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] += background;
                if(trend)
                {
                    mu[i] += slope*(t[i]-data.get_t_middle());
                }
                if (obs_after_HARPS_fibers)
                {
                    if (i >= data.index_fibers) mu[i] += fiber_offset;
                }
            }
        }
    }
return logH;
}


double RVmodel::log_likelihood() const
{
    double logL = 0.;
    double logLikelihoods = 0.;
    auto data = Data::get_instance();
    int N = data.N();
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    if(GP)
    {
        if(RN)
        {
        vector<double> y;
        const vector<double>& rv = data.get_rv();
        const vector<double>& fwhm = data.get_fwhm();
        const vector<double>& bis = data.get_bis();
        const vector<double>& rhk = data.get_rhk();
        const vector<double>& sig = data.get_sig();
        const vector<double>& t = data.get_t();
            for(int i=0; i<4; i++)
            {
                /* residual vector (observed y minus model y) */
                VectorXd residual(t.size());
                if(i==0)
                    y = rv;
                    for(size_t j=0; j<t.size(); j++)
                        residual(j) = y[j]-mu[j];
                if(i==1)
                    y = fwhm;
                    for(size_t j=0; j<t.size(); j++)
                        residual(j) = y[j];
                if(i==2)
                    y = bis;
                    for(size_t j=0; j<t.size(); j++)
                        residual(j) = y[j];
                if(i==3)
                    y = rhk;
                    for(size_t j=0; j<t.size(); j++)
                        residual(j) = y[j];

                /* perform the cholesky decomposition of C */
                Eigen::LLT<Eigen::MatrixXd> cholesky = Cs[i].llt();
                /* get the lower triangular matrix L */
                MatrixXd L = cholesky.matrixL();

                double logDeterminant = 0.;
                for(size_t j=0; j<t.size(); j++)
                    logDeterminant += 2.*log(L(j,j));

                VectorXd solution = cholesky.solve(residual);

                /* y*solution */
                double exponent = 0.;
                for(size_t j=0; j<t.size(); j++)
                    exponent += residual(j)*solution(j);

                logLikelihoods = -0.5*y.size()*log(2*M_PI)
                        - 0.5*logDeterminant - 0.5*exponent;
                logLikelihoods += logLikelihoods;
            }
        logL = logLikelihoods;
        }
        else
        {
            const vector<double>& y = data.get_y();
            /* The following code calculates the log likelihood in the case of a GP model */
            /* residual vector (observed y minus model y) */
            VectorXd residual(y.size());
            for(size_t i=0; i<y.size(); i++)
                residual(i) = y[i] - mu[i];

            /* perform the cholesky decomposition of C */
            Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();
            /* get the lower triangular matrix L */
            MatrixXd L = cholesky.matrixL();

            double logDeterminant = 0.;
            for(size_t i=0; i<y.size(); i++)
                logDeterminant += 2.*log(L(i,i));

            VectorXd solution = cholesky.solve(residual);

            /* y*solution */
            double exponent = 0.;
            for(size_t i=0; i<y.size(); i++)
                exponent += residual(i)*solution(i);

            logL = -0.5*y.size()*log(2*M_PI)
                    - 0.5*logDeterminant - 0.5*exponent;
        }
    }

    else
    {
        /* The following code calculates the log likelihood 
        in the case of a Gaussian likelihood */
        double var;
        const vector<double>& y = data.get_y();
        const vector<double>& sig = data.get_sig();
        for(size_t i=0; i<y.size(); i++)
        {
            var = sig[i]*sig[i] + extra_sigma*extra_sigma;
            logL += - halflog2pi - 0.5*log(var)
                    - 0.5*(pow(y[i] - mu[i], 2)/var);
        }
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Likelihood took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

    if(std::isnan(logL) || std::isinf(logL))
    {
        logL = std::numeric_limits<double>::infinity();
    }
    return logL;
}

void RVmodel::print(std::ostream& out) const
{
    /* output precision */
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    out<<extra_sigma<<'\t';
    
    if(trend)
        out<<slope<<'\t';

    if (obs_after_HARPS_fibers)
        out<<fiber_offset<<'\t';

    if(GP)
    {
        if(RN)
        {
        /* We are going to need to check the kernels used and make out of all the
        parameters that are in the node_priors and weight_priors */
        
        
        }
        else
        {
            out<<eta1<<'\t'<<eta2<<'\t'<<eta3<<'\t'<<eta4<<'\t';
        }
    }

  
    planets.print(out);

    out<<' '<<staleness<<' ';
    out<<background;
}

string RVmodel::description() const
{
    string desc;
    
    desc += "extra_sigma\t";

    if(trend)
        desc += "slope\t";
    if (obs_after_HARPS_fibers)
        desc += "fiber_offset\t";
    if(GP)
    {
        if(RN)
        {
        /* We are going to need to check the kernels used and make out of all parameters  */
        
        }
        else
        {
            desc += "eta1\teta2\teta3\teta4\t";
        }
    }



    desc += "ndim\tmaxNp\t";
    if(hyperpriors)
        desc += "muP\twP\tmuK\t";

    desc += "Np\t";

    if (planets.get_max_num_components()>0)
        desc += "P\tK\tphi\tecc\tw\t";

    desc += "staleness\tvsys";

    return desc;
}


void RVmodel::save_setup() {
    /* save the options of the current model in a INI file */
    std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "obs_after_HARPS_fibers: " << obs_after_HARPS_fibers << endl;
    fout << "GP: " << GP << endl;
    fout << "RN: " << RN << endl;
    fout << "hyperpriors: " << hyperpriors << endl;
    fout << "trend: " << trend << endl;
    fout << endl;
    fout << "file: " << Data::get_instance().datafile << endl;
    fout << "units: " << Data::get_instance().dataunits << endl;
    fout << "skip: " << Data::get_instance().dataskip << endl;

    fout.close();
}


/*
    Calculates the eccentric anomaly at time t by solving Kepler's equation.
    See "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006

    @param t the time at which to calculate the eccentric anomaly.
    @param period the orbital period of the planet
    @param ecc the eccentricity of the orbit
    @param t_peri time of periastron passage
    @return eccentric anomaly.
*/
double RVmodel::ecc_anomaly(double t, double period, double ecc, double time_peri)
{
    double tol;
    if (ecc < 0.8) tol = 1e-14;
    else tol = 1e-13;

    double n = 2.*M_PI/period;  // mean motion
    double M = n*(t - time_peri);  // mean anomaly
    double Mnorm = fmod(M, 2.*M_PI);
    double E0 = keplerstart3(ecc, Mnorm);
    double dE = tol + 1;
    double E;
    int count = 0;
    while (dE > tol)
    {
        E = E0 - eps3(ecc, Mnorm, E0);
        dE = abs(E-E0);
        E0 = E;
        count++;
        /* failed to converge, this only happens for nearly parabolic orbits */
        if (count == 100) break;
    }
    return E;
}


/*
    Provides a starting value to solve Kepler's equation.
    See "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006

    @param e the eccentricity of the orbit
    @param M mean anomaly (in radians)
    @return starting value for the eccentric anomaly.
*/
double RVmodel::keplerstart3(double e, double M)
{
    double t34 = e*e;
    double t35 = e*t34;
    double t33 = cos(M);
    return M + (-0.5*t35 + e + (t34 + 1.5*t33*t35)*t33)*sin(M);
}


/*
    An iteration (correction) method to solve Kepler's equation.
    See "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006

    @param e the eccentricity of the orbit
    @param M mean anomaly (in radians)
    @param x starting value for the eccentric anomaly
    @return corrected value for the eccentric anomaly
*/
double RVmodel::eps3(double e, double M, double x)
{
    double t1 = cos(x);
    double t2 = -1 + e*t1;
    double t3 = sin(x);
    double t4 = e*t3;
    double t5 = -x + t4 + M;
    double t6 = t5/(0.5*t5*t4/t2+t2);

    return t5/((0.5*t3 - 1/6*t1*t6)*e*t6+t2);
}



/*
    Calculates the true anomaly at time t.
    See Eq. 2.6 of The Exoplanet Handbook, Perryman 2010

    @param t the time at which to calculate the true anomaly.
    @param period the orbital period of the planet
    @param ecc the eccentricity of the orbit
    @param t_peri time of periastron passage
    @return true anomaly.
*/
double RVmodel::true_anomaly(double t, double period, double ecc, double t_peri)
{
    double E = ecc_anomaly(t, period, ecc, t_peri);
    double f = acos( (cos(E)-ecc)/( 1-ecc*cos(E) ) );
    /* acos gives the principal values ie [0:PI]
    when E goes above PI we need another condition */
    if(E>M_PI)
      f=2*M_PI-f;

    return f;
}
