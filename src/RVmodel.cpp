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
#include "Means.h"

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
extern ContinuousDistribution *constant_weight; //start of kernel priors
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
extern ContinuousDistribution *jitter_prior; //jitters
extern ContinuousDistribution *const_mean; //start of mean priors
extern ContinuousDistribution *linear_slope;
extern ContinuousDistribution *linear_intercept;
extern ContinuousDistribution *parabolic_quadcoeff;
extern ContinuousDistribution *parabolic_lincoeff;
extern ContinuousDistribution *parabolic_free;
extern ContinuousDistribution *cubic_cubcoeff;
extern ContinuousDistribution *cubic_quadcoeff;
extern ContinuousDistribution *cubic_lincoeff;
extern ContinuousDistribution *cubic_free;
extern ContinuousDistribution *sine_amp;
extern ContinuousDistribution *sine_freq;
extern ContinuousDistribution *sine_phase;

double nprior1, nprior2, nprior3, nprior4;
double wprior1, wprior2, wprior3, wprior4;
double jitter1, jitter2, jitter3, jitter4;
double mean1, mean2, mean3, mean4;

/* from the offsets determined by Lo Curto et al. 2015 (only FGK stars)
mean, std = 14.641789473684208, 2.7783035258938971 */
Gaussian *fiber_offset_prior = new Gaussian(15., 3.);
//Uniform *fiber_offset_prior = new Uniform(0., 50.);  // old 

const double halflog2pi = 0.5*log(2.*M_PI);

/*  The GP priors are eta1, eta2, eta3, and eta4
    For the GPRN the priors are nprior1, nprior2, nprior3, nprior4, and nprior5 
for the nodes and wprior1, wprior2, wprior3, wprior4, and wprior5 for the 
weigths */
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

    /* Working with a GP; GP=true and RN=false */
    if(GP & !RN)
    {
        eta1 = exp(log_eta1_prior->generate(rng));  // m/s
        eta2 = exp(log_eta2_prior->generate(rng));  // days
        eta3 = eta3_prior->generate(rng);           // days
        eta4 = exp(log_eta4_prior->generate(rng));
    }

    /* Working with a GPRN; GP=true and RN=true */
    if(RN)
    {
        /* Generate priors accordingly to the kernels in use */
        n_size = GPRN::get_instance().node.size(); //number of nodes
        w_size = 4 * n_size; //number of weights

        /* dealing with the jitters first*/
        jitter1 = jitter_prior->generate(rng);
        jitter2 = jitter_prior->generate(rng);
        jitter3 = jitter_prior->generate(rng);
        jitter4 = jitter_prior->generate(rng);
        jitter_priors = {jitter1, jitter2, jitter3, jitter4};

        /* dealing with the means */
        for(int i=0; i<3; i++)
        {
            if(GPRN::get_instance().mean[i] == "C")
            {
                mean1 = const_mean->generate(rng);
                mean_priors[i] = {mean1};
                mean_type[i] = "C";
            }
            if(GPRN::get_instance().mean[i] == "L")
            {
                mean1 = linear_slope->generate(rng);
                mean2 = linear_intercept->generate(rng);
                mean_priors[i] = {mean1, mean2};
                mean_type[i] = "L";
            }
            if(GPRN::get_instance().mean[i] == "P")
            {
                mean1 = parabolic_quadcoeff->generate(rng);
                mean2 = parabolic_lincoeff->generate(rng);
                mean3 = parabolic_free->generate(rng);
                mean_priors[i] = {mean1, mean2, mean3};
                mean_type[i] = "L";
            }
            if(GPRN::get_instance().mean[i] == "CUB")
            {
                mean1 = cubic_cubcoeff->generate(rng);
                mean2 = cubic_quadcoeff->generate(rng);
                mean3 = cubic_lincoeff->generate(rng);
                mean4 = cubic_free->generate(rng);
                mean_priors[i] = {mean1, mean2, mean3, mean4};
                mean_type[i] = "CUB";
            }
            if(GPRN::get_instance().mean[i] == "SIN")
            {
                mean1 = sine_amp->generate(rng);
                mean2 = sine_freq->generate(rng);
                mean3 = sine_phase->generate(rng);
                mean_priors[i] = {mean1, mean2, mean3};
                mean_type[i] = "SIN";
            }
            if(GPRN::get_instance().mean[i] == "None")
            {
                mean_priors[i] = {0};
                mean_type[i] = "None";
            }
        }

        /* dealing with the nodes */
        for(int i=0; i<n_size; i++)
        {
            if(GPRN::get_instance().node[i] == "C")
            {
                nprior1 = constant_prior->generate(rng);
                node_priors[i] = {nprior1};
            }
            if(GPRN::get_instance().node[i] == "SE")
            {
                nprior1 = se_ell->generate(rng);
                node_priors[i] = {nprior1};
            }
            if(GPRN::get_instance().node[i] == "P")
            {
                nprior1 = per_ell->generate(rng);
                nprior2 = per_period->generate(rng);
                node_priors[i] = {nprior1, nprior2};
            }
            if(GPRN::get_instance().node[i] == "QP")
            {
                nprior1 = quasi_elle->generate(rng);
                nprior2 = quasi_period->generate(rng);
                nprior3 = quasi_ellp->generate(rng);
                node_priors[i] = {nprior1, nprior2, nprior3};
                /* printing stuff */
                //cout << "node params = " << nprior1 << " ";
                //cout << nprior2 << " " << nprior3 << endl;
            }
            if(GPRN::get_instance().node[i] == "RQ")
            {
                nprior1 = ratq_alpha->generate(rng);
                nprior2 = ratq_ell->generate(rng);
                node_priors[i] = {nprior1, nprior2};
            }
            if(GPRN::get_instance().node[i] == "COS")
            {
                nprior1 = cos_period->generate(rng);
                node_priors[i] = {nprior1};
            }
            if(GPRN::get_instance().node[i] == "EXP")
            {
                nprior1 = exp_ell->generate(rng);
                node_priors[i] = {nprior1};
            }
            if(GPRN::get_instance().node[i] == "M32")
            {
                nprior1 = m32_ell->generate(rng);
                node_priors[i] = {nprior1};
            }
            if(GPRN::get_instance().node[i] == "M52")
            {
                nprior1 = m52_ell->generate(rng);
                node_priors[i] = {nprior1};
            }
        }

        /* dealing with the weights */
        if(GPRN::get_instance().weight[0] == "C")
        {
            for(int j=0; j<w_size; j++)
            {
                wprior1 = constant_weight->generate(rng);
                weight_priors[j] = {wprior1};
                /* printing stuff */
                //cout << "weights params = " << wprior1 << " ";
            }
            //cout << endl;
        }
        if(GPRN::get_instance().weight[0] == "SE")
        {
            wprior2 = se_ell->generate(rng);
            for(int j=0; j<w_size; j++)
            {
                wprior1 = se_weight->generate(rng);
                weight_priors[j] = {wprior1, wprior2};
            }
        }
        if(GPRN::get_instance().weight[0] == "P")
        {
            wprior2 = per_ell->generate(rng);
            wprior3 = per_period->generate(rng);
            for(int j=0; j<w_size; j++)
            {
                wprior1 = per_weight->generate(rng);
                weight_priors[j] = {wprior1, wprior2, wprior3};
            }
        }
        if(GPRN::get_instance().weight[0] == "QP")
        {
            wprior2 = quasi_elle->generate(rng);
            wprior3 = quasi_period->generate(rng);
            wprior4 = quasi_ellp->generate(rng);
            for(int j=0; j<w_size; j++)
            {
                wprior1 = quasi_weight->generate(rng);
                weight_priors[j] = {wprior1, wprior2, wprior3, wprior4};
            }
        }
        if(GPRN::get_instance().weight[0] == "RQ")
        {
            wprior2 = ratq_alpha->generate(rng);
            wprior3 = ratq_ell->generate(rng);
            for(int j=0; j<w_size; j++)
            {
                wprior1 = ratq_weight->generate(rng);
                weight_priors[j] = {wprior1, wprior2, wprior3};
            }
        }
        if(GPRN::get_instance().weight[0] == "COS")
        {
            wprior2 = cos_period->generate(rng);
            for(int j=0; j<w_size; j++)
            {
                wprior1 = cos_weight->generate(rng);
                weight_priors[j] = {wprior1, wprior2};
            }
        }
        if(GPRN::get_instance().weight[0] == "EXP")
        {
            wprior2 = exp_ell->generate(rng);
            for(int j=0; j<w_size; j++)
            {
                wprior1 = exp_weight->generate(rng);
                weight_priors[j] = {wprior1, wprior2};
            }
        }
        if(GPRN::get_instance().weight[0] == "M32")
        {
            wprior2 = m32_ell->generate(rng);
            for(int j=0; j<w_size; j++)
            {
                wprior1 = m32_weight->generate(rng);
                weight_priors[j] = {wprior1, wprior2};
            }
        }
        if(GPRN::get_instance().weight[0] == "M52")
        {
            wprior2 = m52_ell->generate(rng);
            for(int j=0; j<w_size; j++)
            {
                wprior1 = m52_weight->generate(rng);
                weight_priors[j] = {wprior1, wprior2};
            }
        }
    }
    /* calculation of the mean and covariance kernel */
    calculate_mu();
    if(GP) calculate_C();
}


void RVmodel::calculate_C()
{
    /* if we want a GPRN the RN is set to true; GP=true and RN=true*/
    if(RN)
    {
        /* data */
        auto data = Data::get_instance();
        const vector<double>& t = data.get_t();
        const vector<double>& sig = data.get_sig();
        int N = data.get_t().size();
        
        Cs = GPRN::get_instance().matrixCalculation(node_priors, weight_priors, 
                                                    jitter_priors, extra_sigma);
        C = Cs[0]*0;    //I dont know why but it doesnt run if we dont define C
        
    }
    /* otherwise we just want a GP; GP=true and RN=false*/
    else
    {
        /* data */
        auto data = Data::get_instance();
        const vector<double>& t = data.get_t();
        const vector<double>& sig = data.get_sig();
        int N = data.get_t().size();
        
        /* Quasi periodic kernel */
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
    /* just updating (adding) planets */
    else
        staleness++;

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now(); // start timing
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

    /* using a simple GP; GP=true and RN=false */
    if(GP & !RN)
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
            // propose new fiber offset
            if (obs_after_HARPS_fibers)
            {
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
            /* dealing with the nodes */
            if(rng.rand() > 0.5)
            {
                for(int i=0; i<n_size; i++)
                {
                    if(GPRN::get_instance().node[i] == "C")
                    {
                        constant_prior->perturb(nprior1, rng);
                        node_priors[i] = {nprior1};
                    }
                    if(GPRN::get_instance().node[i] == "SE")
                    {
                        se_ell->perturb(nprior1, rng);
                        node_priors[i] = {nprior1};
                    }
                    if(GPRN::get_instance().node[i] == "P")
                    {
                        per_ell->perturb(nprior1, rng);
                        per_period->perturb(nprior2, rng);
                        node_priors[i] = {nprior1, nprior2};
                    }
                    if(GPRN::get_instance().node[i] == "QP")
                    {
                        quasi_elle->perturb(nprior1, rng);
                        quasi_period->perturb(nprior2, rng);
                        quasi_ellp->perturb(nprior3, rng);
                        node_priors[i] = {nprior1, nprior2, nprior3};
                        /* printing stuff */
                        //cout << "node params = " << nprior1 << " ";
                        //cout << nprior2 << " " << nprior3 << endl;
                    }
                    if(GPRN::get_instance().node[i] == "RQ")
                    {
                        ratq_alpha->perturb(nprior1, rng);
                        ratq_ell->perturb(nprior2, rng);
                        node_priors[i] = {nprior1, nprior2};
                    }
                    if(GPRN::get_instance().node[i] == "COS")
                    {
                        cos_period->perturb(nprior1, rng);
                        node_priors[i] = {nprior1};
                    }
                    if(GPRN::get_instance().node[i] == "EXP")
                    {
                        exp_ell->perturb(nprior1, rng);
                        node_priors[i] = {nprior1};
                    }
                    if(GPRN::get_instance().node[i] == "M32")
                    {
                        m32_ell->perturb(nprior1, rng);
                        node_priors[i] = {nprior1};
                    }
                    if(GPRN::get_instance().node[i] == "M52")
                    {
                        m52_ell->perturb(nprior1, rng);
                        node_priors[i] = {nprior1};
                    }
                }
            }
            /* dealing with the weights */
            if(rng.rand() < 0.5)
            {
                if(GPRN::get_instance().weight[0] == "C")
                {
                    for(int j=0; j<w_size; j++)
                    {
                        constant_weight->perturb(wprior1, rng);
                        if(rng.rand() > 0.5)
                        {
                            weight_priors[j] = {wprior1};
                        }
                    }
                }
                if(GPRN::get_instance().weight[0] == "SE")
                {
                    /* All weights sahre the same hyperparameters, except 
                    for the amplitude, that's why some are outside and 
                    others outside the for loop */
                    se_ell->perturb(wprior2, rng);
                    for(int j=0; j<w_size; j++)
                    {
                        se_weight->perturb(wprior1, rng);
                        weight_priors[j] = {wprior1, wprior2};
                    }
                }
                if(GPRN::get_instance().weight[0] == "P")
                {
                    per_ell->perturb(wprior2, rng);
                    per_period->perturb(wprior3, rng);
                    for(int j=0; j<w_size; j++)
                    {
                        per_weight->perturb(wprior1, rng);
                        weight_priors[j] = {wprior1, wprior2, wprior3};
                    }
                }
                if(GPRN::get_instance().weight[0] == "QP")
                {
                    quasi_elle->perturb(wprior2, rng);
                    quasi_period->perturb(wprior3, rng);
                    quasi_ellp->perturb(wprior4, rng);
                    for(int j=0; j<w_size; j++)
                    {
                        quasi_weight->perturb(wprior1, rng);
                        weight_priors[j] = {wprior1, wprior2, wprior3, wprior4};
                    }
                }
                if(GPRN::get_instance().weight[0] == "RQ")
                {
                    ratq_alpha->perturb(wprior2, rng);
                    ratq_ell->perturb(wprior3, rng);
                    for(int j=0; j<w_size; j++)
                    {
                        ratq_weight->perturb(wprior1, rng);
                        weight_priors[j] = {wprior1, wprior2, wprior3};
                    }
                }
                if(GPRN::get_instance().weight[0] == "COS")
                {
                    cos_period->perturb(wprior2, rng);
                    for(int j=0; j<w_size; j++)
                    {
                        cos_weight->perturb(wprior1, rng);
                        weight_priors[j] = {wprior1, wprior2};
                    }
                }
                if(GPRN::get_instance().weight[0] == "EXP")
                {
                    exp_ell->perturb(wprior2, rng);
                    for(int j=0; j<w_size; j++)
                    {
                        exp_weight->perturb(wprior1, rng);
                        weight_priors[j] = {wprior1, wprior2};
                    }
                }
                if(GPRN::get_instance().weight[0] == "M32")
                {
                    m32_ell->perturb(wprior2, rng);
                    for(int j=0; j<w_size; j++)
                    {
                        m32_weight->perturb(wprior1, rng);
                        weight_priors[j] = {wprior1, wprior2};
                    }
                }
                if(GPRN::get_instance().weight[0] == "M52")
                {
                    m52_ell->perturb(wprior2, rng);
                    for(int j=0; j<w_size; j++)
                    {
                        m52_weight->perturb(wprior1, rng);
                        weight_priors[j] = {wprior1, wprior2};
                    }
                }
            }
            calculate_C();
        }
        if(rng.rand() <= 0.5)
        {
            Jprior->perturb(extra_sigma, rng);
            /* GPRN jitters */
            jitter_prior->perturb(jitter1, rng);
            jitter_priors[0] = jitter1;
            jitter_prior->perturb(jitter2, rng);
            jitter_priors[1] = jitter2;
            jitter_prior->perturb(jitter3, rng);
            jitter_priors[2] = jitter3;
            jitter_prior->perturb(jitter4, rng);
            jitter_priors[3] = jitter4;
            calculate_C();
        }
        else
        {
            /* GPRN means */
            for(int i=0; i<3; i++)
            {
                if(GPRN::get_instance().mean[i] == "C")
                {
                    const_mean->perturb(mean1, rng);
                    mean_priors[i] = {mean1};
                }
                if(GPRN::get_instance().mean[i] == "L")
                {
                    linear_slope->perturb(mean1, rng);
                    linear_intercept->perturb(mean2, rng);
                    mean_priors[i] = {mean1, mean2};
                }
                if(GPRN::get_instance().mean[i] == "P")
                {
                    parabolic_quadcoeff->perturb(mean1, rng);
                    parabolic_lincoeff->perturb(mean2, rng);
                    parabolic_free->perturb(mean3, rng);
                    mean_priors[i] = {mean1, mean2, mean3};
                }
                if(GPRN::get_instance().mean[i] == "CUB")
                {
                    cubic_cubcoeff->perturb(mean1, rng);
                    cubic_quadcoeff->perturb(mean2, rng);
                    cubic_lincoeff->perturb(mean3, rng);
                    cubic_free->perturb(mean4, rng);
                    mean_priors[i] = {mean1, mean2, mean3, mean4};
                }
                if(GPRN::get_instance().mean[i] == "SIN")
                {
                    sine_amp->perturb(mean1, rng);
                    sine_freq->perturb(mean2, rng);
                    sine_phase->perturb(mean3, rng);
                    mean_priors[i] = {mean1, mean2, mean3};
                }
                if(GPRN::get_instance().mean[i] == "None")
                {
                    mean_priors[i] = {0};
                }
            }
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
            // propose new fiber offset
            if (obs_after_HARPS_fibers)
            {
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
    /* With no GPs, it's just a sum of Keplerians*/
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
    auto begin = std::chrono::high_resolution_clock::now(); // start timing
    #endif

    /* using a standard GP; GP=true and RN=true */
    if(GP & !RN)
    {
        const vector<double>& y = data.get_y();
        /* Log likelihood in the case of a GP model */
        /* residual vector (observed y minus model y) */
        VectorXd residual(y.size());
        for(size_t i=0; i<y.size(); i++)
        {
            residual(i) = y[i] - mu[i];
        }

        /* perform the cholesky decomposition of C */
        Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();
        /* get the lower triangular matrix L */
        MatrixXd L = cholesky.matrixL();

        double logDeterminant = 0.;
        for(size_t i=0; i<y.size(); i++)
        {
            logDeterminant += 2.*log(L(i,i));
        }

        VectorXd solution = cholesky.solve(residual);

        /* y*solution */
        double exponent = 0.;
        for(size_t i=0; i<y.size(); i++)
        {
            exponent += residual(i)*solution(i);
        }
        logL = -0.5*y.size()*log(2*M_PI)
                - 0.5*logDeterminant - 0.5*exponent;
    }
    /* using a GPRN; GP=true and RN=true */
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
            {
                y = rv;
                for(size_t j=0; j<t.size(); j++)
                    residual(j) = y[j]-mu[j];
            }
            if(i==1)
            {
                y = fwhm;
                for(size_t j=0; j<t.size(); j++)
                    residual(j) = y[j] - Means::get_instance().meanCalc(mean_type[0], mean_priors[0], t[j]);
            }
            if(i==2)
            {
                y = bis;
                for(size_t j=0; j<t.size(); j++)
                    residual(j) = y[j] - Means::get_instance().meanCalc(mean_type[1], mean_priors[1], t[j]);
            }
            if(i==3)
            {
                y = rhk;
                for(size_t j=0; j<t.size(); j++)
                {
                    residual(j) = y[j] - Means::get_instance().meanCalc(mean_type[2], mean_priors[2], t[j]);
                }
            }
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
        /* The following code calculates the log likelihood 
        in the case of a Gaussian likelihood */
        double var;
        const vector<double>& y = data.get_y();
        const vector<double>& sig = data.get_sig();
        for(size_t i=0; i<y.size(); i++)
        {
            var = sig[i]*sig[i] + extra_sigma*extra_sigma;
            logL += - halflog2pi - 0.5*log(var) - 0.5*(pow(y[i] - mu[i], 2)/var);
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
        out<<eta1<<'\t'<<eta2<<'\t'<<eta3<<'\t'<<eta4<<'\t';
    }

    planets.print(out);
    out<<' '<<staleness<<' ';
    out<<background << '\t';

    if(RN)
    {
        std::vector<double> gprn_outputs; //to put all parameters inside it
        /* Lets start by inserting the nodes parameters */
        for(int i=0; i<node_priors.size(); i++)
        {
            for(int j=0; j<node_priors[i].size(); j++)
                {
                gprn_outputs.push_back(node_priors[i][j]);
                }
        }
        /* Then we add the weights parameters */
        for(int i=0; i<weight_priors.size(); i++)
        {
            for(int j=0; j<weight_priors[i].size(); j++)
                gprn_outputs.push_back(weight_priors[i][j]);
        }
        /* And for last the jitter */
        for(int i=0; i<4; i++)
        {
            gprn_outputs.push_back(jitter_priors[i]);
        }
        
        /* Finally we print them all in the file */
        for (int ii = 0; ii < gprn_outputs.size(); ii++)
        {
            out << gprn_outputs[ii] << '\t';
        }
    }
}

string RVmodel::description() const
{
    std::string desc;
    
    desc += "extra_sigma\t";

    if(trend)
        desc += "slope\t";
    if (obs_after_HARPS_fibers)
        desc += "fiber_offset\t";

    if(GP)
    {
        desc += "eta1\teta2\teta3\teta4\t";
    }
    desc += "ndim\tmaxNp\t";
    if(hyperpriors)
        desc += "muP\twP\tmuK\t";

    desc += "Np\t";

    if (planets.get_max_num_components()>0)
        desc += "P\tK\tphi\tecc\tw\t";

    desc += "staleness\tvsys\t";

    if(RN)
    {
        /* first we name our node babies */
        for(int i=0; i<node_priors.size(); i++)
        {
            for(int j=0; j<node_priors[i].size(); j++)
                {
                std::string node_header = "node";
                node_header += std::to_string(i);
                node_header += "_";
                node_header += std::to_string(j);
                desc += node_header;
                desc += '\t';
                }
        }
        /* Then we name the weights babies */
        for(int i=0; i<weight_priors.size(); i++)
        {
            for(int j=0; j<weight_priors[i].size(); j++)
                {
                std::string weight_header = "weight";
                weight_header += std::to_string(i);
                weight_header += "_";
                weight_header += std::to_string(j);
                desc += weight_header;
                desc += '\t';
                }
        }
        /* For last are the jitter bastards */
        for(int i=0; i<4; i++)
        {
            std::string jitter_header = "jitter";
            jitter_header += std::to_string(i);
            desc += jitter_header;
            desc += '\t';
        }
    }
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
    if(RN)
    {
        //printing the nodes
        fout << "nodes: ";
        for (auto i = GPRN::get_instance().node.begin(); i != GPRN::get_instance().node.end(); ++i)
            fout << *i << ' ';
        fout << endl;
        //printing just the weight[0] because they are all the same
        fout << "weights: " << GPRN::get_instance().weight[0] << endl;
    }
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
