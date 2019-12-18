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

using namespace std;
using namespace Eigen;
using namespace DNest4;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);



/* set default priors if the user didn't change them */
void RVmodel::setPriors()  // BUG: should be done by only one thread!
{
    auto data = Data::get_instance();

    betaprior = make_prior<Gaussian>(0, 1);
    sigmaMA_prior = make_prior<ModifiedLogUniform>(1.0, 10.);
    tauMA_prior = make_prior<LogUniform>(1, 10);
    
    if (!Cprior)
        Cprior = make_prior<Uniform>(data.get_y_min(), data.get_y_max());

    if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(1.0, 100.);

    if (!slope_prior)
        slope_prior = make_prior<Uniform>( -data.topslope(), data.topslope() );

    if (!offsets_prior)
        offsets_prior = make_prior<Uniform>( -data.get_RV_span(), data.get_RV_span() );

    if (GP) { /* GP parameters */
        if (!log_eta1_prior)
            log_eta1_prior = make_prior<Uniform>(-5, 5);
        if (!eta2_prior)
            eta2_prior = make_prior<LogUniform>(1, 100);
        if (!eta3_prior)
            eta3_prior = make_prior<Uniform>(10, 40);
        if (!log_eta4_prior)
            log_eta4_prior = make_prior<Uniform>(-1, 1);
    }

    if (!fiber_offset_prior)
        fiber_offset_prior = make_prior<Uniform>(0, 50);
        // fiber_offset_prior = make_prior<Gaussian>(15., 3.);
    

    if (known_object) { // KO mode!
        if (!KO_Pprior || !KO_Kprior || !KO_eprior || !KO_phiprior || !KO_wprior)
            throw std::logic_error("When known_object=true, please set all priors: KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior");
    }

}


void RVmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    background = Cprior->generate(rng);

    if(multi_instrument)
    {
        for(int i=0; i<offsets.size(); i++)
            offsets[i] = offsets_prior->generate(rng);
        for(int i=0; i<jitters.size(); i++)
            jitters[i] = Jprior->generate(rng);
    }
    else
    {
        extra_sigma = Jprior->generate(rng);
    }


    if(obs_after_HARPS_fibers)
        fiber_offset = fiber_offset_prior->generate(rng);

    if(trend)
        slope = slope_prior->generate(rng);

    if(GP)
    {
        eta1 = exp(log_eta1_prior->generate(rng)); // m/s

        eta2 = eta2_prior->generate(rng); // days

        eta3 = eta3_prior->generate(rng); // days

        eta4 = exp(log_eta4_prior->generate(rng));
    }

    if(MA)
    {
        sigmaMA = sigmaMA_prior->generate(rng);
        tauMA = tauMA_prior->generate(rng);
    }

    auto data = Data::get_instance();
    if (data.indicator_correlations)
    {
        for (unsigned i=0; i<data.number_indicators; i++)
            betas[i] = betaprior->generate(rng);
    }

    if (known_object) { // KO mode!
        KO_P = KO_Pprior->generate(rng);
        KO_K = KO_Kprior->generate(rng);
        KO_e = KO_eprior->generate(rng);
        KO_phi = KO_phiprior->generate(rng);
        KO_w = KO_wprior->generate(rng);
    }

    calculate_mu();

    if(GP) calculate_C();

}

/**
 * @brief Fill the GP covariance matrix.
 * 
*/
void RVmodel::calculate_C()
{
    // Get the data
    auto data = Data::get_instance();
    const vector<double>& t = data.get_t();
    const vector<double>& sig = data.get_sig();
    const vector<int>& obsi = data.get_obsi();
    int N = data.N();
    double jit;

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = eta1*eta1*exp(-0.5*pow((t[i] - t[j])/eta2, 2)
                        -2.0*pow(sin(M_PI*(t[i] - t[j])/eta3)/eta4, 2) );

            if(i==j)
            {
                if (multi_instrument)
                {
                    jit = jitters[obsi[i]-1];
                    C(i, j) += sig[i]*sig[i] + jit*jit;
                }
                else
                {
                    C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
                }
            }
            else
            {
                C(j, i) = C(i, j);
            }
        }
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "GP build matrix: ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    cout << " ns" << "\t"; // << std::endl;
    #endif
}

/**
 * @brief Calculate the full RV model
 * 
*/
void RVmodel::calculate_mu()
{
    auto data = Data::get_instance();
    // Get the times from the data
    const vector<double>& t = data.get_t();
    // only really needed if multi_instrument
    const vector<int>& obsi = data.get_obsi();
    auto actind = data.get_actind();

    // Update or from scratch?
    bool update = (planets.get_added().size() < planets.get_components().size()) &&
            (staleness <= 10);

    // Get the components
    const vector< vector<double> >& components = (update)?(planets.get_added()):
                (planets.get_components());
    // at this point, components has:
    //  if updating: only the added planets' parameters
    //  if from scratch: all the planets' parameters

    // Zero the signal
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

        if(multi_instrument)
        {
            for(size_t j=0; j<offsets.size(); j++)
            {
                for(size_t i=0; i<t.size(); i++)
                {
                    if (obsi[i] == j+1) { mu[i] += offsets[j]; }
                }
            }
        }

        if(obs_after_HARPS_fibers)
        {
            for(size_t i=data.index_fibers; i<t.size(); i++)
            {
                mu[i] += fiber_offset;
            }
        }

        if(data.indicator_correlations)
        {
            for(size_t i=0; i<t.size(); i++)
            {
                for(size_t j = 0; j < data.number_indicators; j++)
                   mu[i] += betas[j] * actind[j][i];
            }   
        }

        if (known_object) { // KO mode!
            add_known_object();
        }
    }
    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    double f, v, ti;
    double P, K, phi, ecc, omega;
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
            f = true_anomaly(ti, P, ecc, data.M0_epoch-(P*phi)/(2.*M_PI));
            v = K*(cos(f+omega) + ecc*cos(omega));
            mu[i] += v;
        }
    }


    if(MA)
    {
        const vector<double>& y = data.get_y();
        for(size_t i=1; i<t.size(); i++) // the loop starts at the 2nd point
        {
            // y[i-1] - mu[i-1] is the residual at the i-1 observation
            mu[i] += sigmaMA * exp(-fabs(t[i-1] - t[i]) / tauMA) * (y[i-1] - mu[i-1]);
        }   
    }



    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}

void RVmodel::remove_known_object()
{
    auto data = Data::get_instance();
    const vector<double>& t = data.get_t();
    double f, v, ti;
    for(size_t i=0; i<t.size(); i++)
    {
        ti = t[i];
        f = true_anomaly(ti, KO_P, KO_e, data.M0_epoch-(KO_P*KO_phi)/(2.*M_PI));
        v = KO_K*(cos(f+KO_w) + KO_e*cos(KO_w));
        mu[i] -= v;
    }
}

void RVmodel::add_known_object()
{
    auto data = Data::get_instance();
    const vector<double>& t = data.get_t();
    double f, v, ti;
    for(size_t i=0; i<t.size(); i++)
    {
        ti = t[i];
        f = true_anomaly(ti, KO_P, KO_e, data.M0_epoch-(KO_P*KO_phi)/(2.*M_PI));
        v = KO_K*(cos(f+KO_w) + KO_e*cos(KO_w));
        mu[i] += v;
    }
}


double RVmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    auto data = Data::get_instance();
    const vector<double>& t = data.get_t();
    const vector<int>& obsi = data.get_obsi();
    auto actind = data.get_actind();
    double logH = 0.;

    if(GP)
    {
        if(rng.rand() <= 0.5) // perturb planet parameters
        {
            logH += planets.perturb(rng);
            planets.consolidate_diff();
            calculate_mu();
        }
        else if(rng.rand() <= 0.5) // perturb GP parameters
        {
            if(rng.rand() <= 0.25)
            {
                log_eta1 = log(eta1);
                log_eta1_prior->perturb(log_eta1, rng);
                eta1 = exp(log_eta1);
            }
            else if(rng.rand() <= 0.33330)
            {
                // log_eta2 = log(eta2);
                // log_eta2_prior->perturb(log_eta2, rng);
                // eta2 = exp(log_eta2);
                eta2_prior->perturb(eta2, rng);
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
        else if(rng.rand() <= 0.5) // perturb jitter(s)
        {
            if(multi_instrument)
            {
                for(int i=0; i<jitters.size(); i++)
                    Jprior->perturb(jitters[i], rng);
            }
            else
            {
                Jprior->perturb(extra_sigma, rng);
            }
            calculate_C();
        }
        else // perturb other parameters: vsys, slope, offsets
        {
            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] -= background;
                if(trend) {
                    mu[i] -= slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] -= offsets[j]; }
                    }
                }
                if (obs_after_HARPS_fibers) {
                    if (i >= data.index_fibers) mu[i] -= fiber_offset;
                }
            }

            Cprior->perturb(background, rng);

            // propose new instrument offsets
            if (multi_instrument){
                for(unsigned j=0; j<offsets.size(); j++)
                    offsets_prior->perturb(offsets[j], rng);
            }

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
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] += offsets[j]; }
                    }
                }
                if (obs_after_HARPS_fibers) {
                    if (i >= data.index_fibers) mu[i] += fiber_offset;
                }
            }
        }

    }


    else if(MA)
    {
        if(rng.rand() <= 0.5) // perturb planet parameters
        {
            logH += planets.perturb(rng);
            planets.consolidate_diff();
            calculate_mu();
        }
        else if(rng.rand() <= 0.5) // perturb MA parameters
        {
            if(rng.rand() <= 0.5)
                sigmaMA_prior->perturb(sigmaMA, rng);
            else
                tauMA_prior->perturb(tauMA, rng);
            
            calculate_mu();
        }
        else if(rng.rand() <= 0.5) // perturb jitter(s)
        {
            if(multi_instrument)
            {
                for(int i=0; i<jitters.size(); i++)
                    Jprior->perturb(jitters[i], rng);
            }
            else
            {
                Jprior->perturb(extra_sigma, rng);
            }
            calculate_C();
        }
        else // perturb other parameters: vsys, slope, offsets
        {
            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] -= background;
                if(trend) {
                    mu[i] -= slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] -= offsets[j]; }
                    }
                }
                if (obs_after_HARPS_fibers) {
                    if (i >= data.index_fibers) mu[i] -= fiber_offset;
                }

                if(data.indicator_correlations) {
                    for(size_t j = 0; j < data.number_indicators; j++){
                        mu[i] -= betas[j] * actind[j][i];
                    }
                }

            }

            Cprior->perturb(background, rng);

            // propose new instrument offsets
            if (multi_instrument){
                for(unsigned j=0; j<offsets.size(); j++)
                    offsets_prior->perturb(offsets[j], rng);
            }

            // propose new fiber offset
            if (obs_after_HARPS_fibers) {
                fiber_offset_prior->perturb(fiber_offset, rng);
            }

            // propose new slope
            if(trend) {
                slope_prior->perturb(slope, rng);
            }

            if(data.indicator_correlations){
                for(size_t j = 0; j < data.number_indicators; j++){
                    betaprior->perturb(betas[j], rng);
                }
            }

            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] += background;
                if(trend) {
                    mu[i] += slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] += offsets[j]; }
                    }
                }
                if (obs_after_HARPS_fibers) {
                    if (i >= data.index_fibers) mu[i] += fiber_offset;
                }

                if(data.indicator_correlations) {
                    for(size_t j = 0; j < data.number_indicators; j++){
                        mu[i] += betas[j]*actind[j][i];
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
            if(multi_instrument)
            {
                for(int i=0; i<jitters.size(); i++)
                    Jprior->perturb(jitters[i], rng);
            }
            else
            {
                Jprior->perturb(extra_sigma, rng);
            }

            if (known_object)
            {
                remove_known_object();
                KO_Pprior->perturb(KO_P, rng);
                KO_Kprior->perturb(KO_K, rng);
                KO_eprior->perturb(KO_e, rng);
                KO_phiprior->perturb(KO_phi, rng);
                KO_wprior->perturb(KO_w, rng);
                add_known_object();
            }
        
        }
        else
        {
            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] -= background;
                if(trend) {
                    mu[i] -= slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] -= offsets[j]; }
                    }
                }
                if (obs_after_HARPS_fibers) {
                    if (i >= data.index_fibers) mu[i] -= fiber_offset;
                }

                if(data.indicator_correlations) {
                    for(size_t j = 0; j < data.number_indicators; j++){
                        mu[i] -= betas[j] * actind[j][i];
                    }
                }
            }

            Cprior->perturb(background, rng);

            // propose new instrument offsets
            if (multi_instrument){
                for(unsigned j=0; j<offsets.size(); j++){
                    offsets_prior->perturb(offsets[j], rng);
                }
            }

            // propose new fiber offset
            if (obs_after_HARPS_fibers) {
                fiber_offset_prior->perturb(fiber_offset, rng);
            }

            // propose new slope
            if(trend) {
                slope_prior->perturb(slope, rng);
            }

            // propose new indicator correlations
            if(data.indicator_correlations){
                for(size_t j = 0; j < data.number_indicators; j++){
                    betaprior->perturb(betas[j], rng);
                }
            }

            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] += background;
                if(trend) {
                    mu[i] += slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] += offsets[j]; }
                    }
                }
                if (obs_after_HARPS_fibers) {
                    if (i >= data.index_fibers) mu[i] += fiber_offset;
                }

                if(data.indicator_correlations) {
                    for(size_t j = 0; j < data.number_indicators; j++){
                        mu[i] += betas[j]*actind[j][i];
                    }
                }
            }
        }
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Perturb took ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6;
    cout << " ms" << std::endl;
    #endif

    return logH;
}

/**
 * Calculate the log-likelihood for the current values of the parameters.
 * 
 * @return double the log-likelihood
*/
double RVmodel::log_likelihood() const
{
    auto data = Data::get_instance();
    int N = data.N();
    auto y = data.get_y();
    auto sig = data.get_sig();
    auto obsi = data.get_obsi();

    double logL = 0.;

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    if(GP)
    {
        /** The following code calculates the log likelihood in the case of a GP model */
        // residual vector (observed y minus model y)
        VectorXd residual(y.size());
        for(size_t i=0; i<y.size(); i++)
            residual(i) = y[i] - mu[i];

        // perform the cholesky decomposition of C
        Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();
        // get the lower triangular matrix L
        MatrixXd L = cholesky.matrixL();

        double logDeterminant = 0.;
        for(size_t i=0; i<y.size(); i++)
            logDeterminant += 2.*log(L(i,i));

        VectorXd solution = cholesky.solve(residual);

        // y*solution
        double exponent = 0.;
        for(size_t i=0; i<y.size(); i++)
            exponent += residual(i)*solution(i);

        logL = -0.5*y.size()*log(2*M_PI)
                - 0.5*logDeterminant - 0.5*exponent;

    }
    else
    {
        // The following code calculates the log likelihood
        // in the case of a t-Student model
        //  for(size_t i=0; i<y.size(); i++)
        //  {
        //      var = sig[i]*sig[i] + extra_sigma*extra_sigma;
        //      logL += gsl_sf_lngamma(0.5*(nu + 1.)) - gsl_sf_lngamma(0.5*nu)
        //          - 0.5*log(M_PI*nu) - 0.5*log(var)
        //          - 0.5*(nu + 1.)*log(1. + pow(y[i] - mu[i], 2)/var/nu);
        //  }

        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var, jit;
        for(size_t i=0; i<N; i++)
        {
            if(multi_instrument)
            {
                jit = jitters[obsi[i]-1];
                var = sig[i]*sig[i] + jit*jit;
            }
            else
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
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    if (multi_instrument)
    {
        for(int j=0; j<jitters.size(); j++)
            out<<jitters[j]<<'\t';
    }
    else
        out<<extra_sigma<<'\t';

    if(trend)
        out<<slope<<'\t';

    if (obs_after_HARPS_fibers)
        out<<fiber_offset<<'\t';

    if (multi_instrument){
        for(int j=0; j<offsets.size(); j++){
            out<<offsets[j]<<'\t';
        }
    }

    auto data = Data::get_instance();
    if(data.indicator_correlations){
        for(int j=0; j<data.number_indicators; j++){
            out<<betas[j]<<'\t';
        }
    }

    if(GP)
        out<<eta1<<'\t'<<eta2<<'\t'<<eta3<<'\t'<<eta4<<'\t';
    
    if(MA)
        out<<sigmaMA<<'\t'<<tauMA<<'\t';

    if(known_object) // KO mode!
        out << KO_P << "\t" << KO_K << "\t" << KO_phi << "\t" << KO_e << "\t" << KO_w << "\t";


    planets.print(out);

    out<<' '<<staleness<<' ';
    out<<background;
}

string RVmodel::description() const
{
    string desc;

    if (multi_instrument)
    {
        for(int j=0; j<jitters.size(); j++)
           desc += "jitter" + std::to_string(j+1) + "   ";
    }
    else
        desc += "extra_sigma   ";

    if(trend)
        desc += "slope   ";

    if (obs_after_HARPS_fibers)
        desc += "fiber_offset   ";

    if (multi_instrument){
        for(unsigned j=0; j<offsets.size(); j++)
            desc += "offset" + std::to_string(j+1) + "   ";
    }

    auto data = Data::get_instance();
    if(data.indicator_correlations){
        for(int j=0; j<data.number_indicators; j++){
            desc += "beta" + std::to_string(j+1) + "   ";
        }
    }


    if(GP)
        desc += "eta1   eta2   eta3   eta4   ";
    
    if(MA)
        desc += "sigmaMA   tauMA   ";

    desc += "ndim   maxNp   ";
    if(hyperpriors)
        desc += "muP   wP   muK   ";

    desc += "Np   ";

    if (planets.get_max_num_components()>0)
        desc += "P   K   phi   ecc   w   ";

    desc += "staleness   vsys";

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void RVmodel::save_setup() {
    auto data = Data::get_instance();
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

	fout << "obs_after_HARPS_fibers: " << obs_after_HARPS_fibers << endl;
    fout << "GP: " << GP << endl;
    fout << "MA: " << MA << endl;
    fout << "hyperpriors: " << hyperpriors << endl;
    fout << "trend: " << trend << endl;
    fout << "multi_instrument: " << multi_instrument << endl;
    fout << "known_object: " << known_object << endl;
    fout << "indicator_correlations: " << data.indicator_correlations << endl;
    fout << "indicators: ";
    for (auto f: data.indicator_names){
        fout << f;
        (f != data.indicator_names.back()) ? fout << ", " : fout << " ";
    }
    fout << endl;

    fout << endl;

    fout << "file: " << data.datafile << endl;
    fout << "units: " << data.dataunits << endl;
    fout << "skip: " << data.dataskip << endl;
    fout << "multi: " << data.datamulti << endl;

    fout << "files: ";
    for (auto f: data.datafiles)
        fout << f << ",";
    fout << endl;
    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Cprior: " << *Cprior << endl;
    fout << "Jprior: " << *Jprior << endl;
    if (trend)
        fout << "slope_prior: " << *slope_prior << endl;
    if (obs_after_HARPS_fibers)
        fout << "fiber_offset_prior: " << *fiber_offset_prior << endl;
    if (multi_instrument)
        fout << "offsets_prior: " << *offsets_prior << endl;

    if (GP){
        fout << endl << "[priors.GP]" << endl;
        fout << "log_eta1_prior: " << *log_eta1_prior << endl;
        // fout << "log_eta2_prior: " << *log_eta2_prior << endl;
        fout << "eta2_prior: " << *eta2_prior << endl;
        fout << "eta3_prior: " << *eta3_prior << endl;
        fout << "log_eta4_prior: " << *log_eta4_prior << endl;
    }


    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        if (hyperpriors){
            fout << endl << "[prior.hyperpriors]" << endl;
            fout << "log_muP_prior: " << *conditional->log_muP_prior << endl;
            fout << "wP_prior: " << *conditional->wP_prior << endl;
            fout << "log_muK_prior: " << *conditional->log_muK_prior << endl;
        }

        fout << endl << "[priors.planets]" << endl;
        fout << "Pprior: " << *conditional->Pprior << endl;
        fout << "Kprior: " << *conditional->Kprior << endl;
        fout << "eprior: " << *conditional->eprior << endl;
        fout << "phiprior: " << *conditional->phiprior << endl;
        fout << "wprior: " << *conditional->wprior << endl;
    }

    if (known_object) {
        fout << endl << "[priors.known_object]" << endl;
        fout << "Pprior: " << *KO_Pprior << endl;
        fout << "Kprior: " << *KO_Kprior << endl;
        fout << "eprior: " << *KO_eprior << endl;
        fout << "phiprior: " << *KO_phiprior << endl;
        fout << "wprior: " << *KO_wprior << endl;
    }

    fout << endl;
	fout.close();
}


/**
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
    double E = M;
    int count = 0;
    while (dE > tol)
    {
        E = E0 - eps3(ecc, Mnorm, E0);
        dE = abs(E-E0);
        E0 = E;
        count++;
        // failed to converge, this only happens for nearly parabolic orbits
        if (count == 100) break;
    }
    return E;
}


/**
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


/**
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



/**
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
    // double E = solve_kepler(t, period, ecc, t_peri);
    double cosE = cos(E);
    double f = acos( (cosE - ecc)/( 1 - ecc*cosE ) );
    // acos gives the principal values ie [0:PI]
    // when E goes above PI we need another condition
    if(E > M_PI)
      f = 2*M_PI - f;

    return f;
}



/// Calculates x-sin(x) and 1-cos(x) to 20 significant digits for x in [0, pi)
/// From github.com/dfm/exoplanet
template <typename T>
inline void RVmodel::sin_cos_reduc (T x, T* SnReduc, T* CsReduc) {
    const T s[] = {T(1)/6, T(1)/20, T(1)/42, T(1)/72, T(1)/110, T(1)/156, T(1)/210, T(1)/272, T(1)/342, T(1)/420};
    const T c[] = {T(0.5), T(1)/12, T(1)/30, T(1)/56, T(1)/90, T(1)/132, T(1)/182, T(1)/240, T(1)/306, T(1)/380};

    bool bigg = x > M_PI_2;
    T u = (bigg) ? M_PI - x : x;
    bool big = u > M_PI_2;
    T v = (big) ? M_PI_2 - u : u;
    T w = v * v;

    T ss = T(1);
    T cc = T(1);
    for (int i = 9; i >= 1; --i) {
        ss = 1 - w * s[i] * ss;
        cc = 1 - w * c[i] * cc;
    }
    ss *= v * w * s[0];
    cc *= w * c[0];

    if (big) {
        *SnReduc = u - 1 + cc;
        *CsReduc = 1 - M_PI_2 + u + ss;
    } else {
        *SnReduc = ss;
        *CsReduc = cc;
    }
    if (bigg) {
        *SnReduc = 2 * x - M_PI + *SnReduc;
        *CsReduc = 2 - *CsReduc;
    }
}


/// Solve Kepler's equation for the eccentric anomaly
/// From github.com/dfm/exoplanet
template <typename T>
inline T RVmodel::solve_kepler (T t, T period, T ecc, T time_peri) {

    const T two_pi = 2 * M_PI;
    T n = two_pi / period;  // mean motion
    T M = n * (t - time_peri);  // mean anomaly

    T M_ref = two_pi * floor(M / two_pi);
    M -= M_ref;

    bool high = M > M_PI;
    if (high) {
        M = two_pi - M;
    }

    T ome = 1.0 - ecc;

    // Get starter
    T M2 = M*M;
    T M3 = M2*M;
    T alpha = (3*M_PI + 1.6*(M_PI-std::abs(M))/(1+ecc) )/(M_PI - 6/M_PI);
    T d = 3*ome + alpha*ecc;
    T r = 3*alpha*d*(d-ome)*M + M3;
    T q = 2*alpha*d*ome - M2;
    T q2 = q*q;
    T w = pow(std::abs(r) + sqrt(q2*q + r*r), 2.0/3);
    T E = (2*r*w/(w*w + w*q + q2) + M) / d;

    // Approximate Mstar = E - e*sin(E) with numerically stability
    T sE, cE;
    sin_cos_reduc (E, &sE, &cE);

    // Refine the starter
    T f_0 = ecc * sE + E * ome - M;
    T f_1 = ecc * cE + ome;
    T f_2 = ecc * (E - sE);
    T f_3 = 1-f_1;
    T d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1);
    T d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3*d_3)*f_3/6);
    T d_42 = d_4*d_4;
    E -= f_0/(f_1 + 0.5*d_4*f_2 + d_4*d_4*f_3/6 - d_42*d_4*f_2/24);

    if (high) {
        E = two_pi - E;
    }

    return E + M_ref;
}


// instantiate for doubles
template void RVmodel::sin_cos_reduc<double>(double, double*, double*);
template double RVmodel::solve_kepler<double>(double, double, double, double);