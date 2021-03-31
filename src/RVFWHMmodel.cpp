#include "RVFWHMmodel.h"
#include "RVConditionalPrior.h"
#include "DNest4.h"
#include "RNG.h"
#include "Utils.h"
#include "Data.h"
#include <cmath>
#include <limits>
#include <fstream>
#include <chrono>
#include <cmath>
#include <time.h>

using namespace std;
using namespace Eigen;
using namespace DNest4;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


/* set default priors if the user didn't change them */
void RVFWHMmodel::setPriors()  // BUG: should be done by only one thread!
{
    auto data = Data::get_instance();

    betaprior = make_prior<Gaussian>(0, 1);
    // sigmaMA_prior = make_prior<ModifiedLogUniform>(1.0, 10.);
    // sigmaMA_prior = make_prior<TruncatedGaussian>(0.0, 1.0, -1.0, 1.0);
    sigmaMA_prior = make_prior<Uniform>(-1, 1);
    tauMA_prior = make_prior<LogUniform>(1, 100);
    
    if (!Vprior)
        Vprior = make_prior<Uniform>(data.get_RV_min(), data.get_RV_max());
    if (!C2prior)
        C2prior = make_prior<Uniform>(data.get_y2_min(), data.get_y2_max());

    if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(min(1.0, 0.1*data.get_max_RV_span()), data.get_max_RV_span());
    if (!J2prior)
        J2prior = make_prior<ModifiedLogUniform>(1.0, data.get_y2_span());

    if (!slope_prior)
        slope_prior = make_prior<Uniform>( -data.topslope(), data.topslope() );

    if (trend){
        if (degree == 0)
            throw std::logic_error("trend=true but degree=0, what gives?");
        if (degree > 3)
            throw std::range_error("can't go higher than 3rd degree trends");
        if (degree >= 1 && !slope_prior)
            slope_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(1)) );
        if (degree >= 2 && !quadr_prior)
            quadr_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(2)) );
        if (degree == 3 && !cubic_prior)
            cubic_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(3)) );
    }

    if (!offsets_prior)
        offsets_prior = make_prior<Uniform>( -data.get_RV_span(), data.get_RV_span() );
    if (!offsets2_prior)
        offsets2_prior = make_prior<Uniform>( -data.get_y2_span(), data.get_y2_span() );

    if (GP) { /* GP parameters */
        // η1
        if (!eta1_1_prior)
            eta1_1_prior = make_prior<ModifiedLogUniform>(1, data.get_RV_span());
        if (!eta1_2_prior)
            eta1_2_prior = make_prior<ModifiedLogUniform>(1, data.get_y2_span());

        // η2
        if (!eta2_1_prior)
            eta2_1_prior = make_prior<LogUniform>(1, data.get_timespan());
        if (!share_eta2){
            if (!eta2_2_prior)
                eta2_2_prior = make_prior<LogUniform>(1, data.get_timespan());
        }

        // η3
        if (!eta3_1_prior)
            eta3_1_prior = make_prior<Uniform>(10, 40);
        if (!share_eta3){
            if (!eta3_2_prior)
                eta3_2_prior = make_prior<Uniform>(10, 40);
        }

        // η4
        if (kernel == standard || kernel == qpc){
            if (!eta4_1_prior)
                eta4_1_prior = make_prior<LogUniform>(0.1, 10);
            if (!share_eta4){
                if (!eta4_2_prior)
                    eta4_2_prior = make_prior<LogUniform>(0.1, 10);
            }   

        }

        // η5
        if (kernel == qpc){
            if (!eta5_1_prior)
                eta5_1_prior = make_prior<ModifiedLogUniform>(1, data.get_RV_span());
            if (!share_eta5){
                if (!eta5_2_prior)
                    eta5_2_prior = make_prior<ModifiedLogUniform>(1, data.get_y2_span());
            }   
        }

    }

    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior)");
        }
    }

    if (studentt)
        nu_prior = make_prior<LogUniform>(2, 1000);

}


void RVFWHMmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    bkg = Vprior->generate(rng);
    bkg2 = C2prior->generate(rng);

    if (multi_instrument)
    {
        // draw instrument offsets for 1st output
        for (size_t i = 0; i < offsets.size() / 2; i++)
            offsets[i] = offsets_prior->generate(rng);
        // and 2nd output
        for (size_t i = offsets.size() / 2; i < offsets.size(); i++)
            offsets[i] = offsets2_prior->generate(rng);

        // draw jitters for 1st output
        for (size_t i = 0; i < jitters.size() / 2; i++)
            jitters[i] = Jprior->generate(rng);
        // and 2nd output
        for (size_t i = jitters.size() / 2; i < jitters.size(); i++)
            jitters[i] = J2prior->generate(rng);
    }
    else
    {
        jitter1 = Jprior->generate(rng);
        jitter2 = J2prior->generate(rng);
    }


    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }

    if(GP)
    {
        eta1_1 = eta1_1_prior->generate(rng); // m/s
        eta1_2 = eta1_2_prior->generate(rng); // [2nd output units]

        eta2_1 = eta2_1_prior->generate(rng); // days
        if (!share_eta2)
            eta2_2 = eta2_2_prior->generate(rng); // days

        eta3_1 = eta3_1_prior->generate(rng); // days
        if (!share_eta3)
            eta3_2 = eta3_2_prior->generate(rng); // days

        if (kernel == standard){
            eta4_1 = exp(eta4_1_prior->generate(rng));
            if (!share_eta4)
                eta4_2 = eta4_2_prior->generate(rng); // days
        }

        if (kernel == qpc){
            eta5_1 = exp(eta5_1_prior->generate(rng));
            if (!share_eta5)
                eta5_2 = eta5_2_prior->generate(rng); // days
        }

    }

    if(MA)
    {
        sigmaMA = sigmaMA_prior->generate(rng);
        tauMA = tauMA_prior->generate(rng);
    }

    auto data = Data::get_instance();

    if (known_object) { // KO mode!
        KO_P.resize(n_known_object);
        KO_K.resize(n_known_object);
        KO_e.resize(n_known_object);
        KO_phi.resize(n_known_object);
        KO_w.resize(n_known_object);

        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_K[i] = KO_Kprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_w[i] = KO_wprior[i]->generate(rng);
        }
    }

    if (studentt)
        nu = nu_prior->generate(rng);


    calculate_mu();
    calculate_mu_2();

    if(GP){
        calculate_C_1();
        calculate_C_2();
    } 

}

/**
 * @brief Fill the GP covariance matrix.
 * 
*/
void RVFWHMmodel::calculate_C_1()
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

    switch (kernel)
    {
    case standard:
        {
            /* This implements the "standard" quasi-periodic kernel, see R&W2006 */
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    C_1(i, j) = eta1_1*eta1_1 * \
                              exp(-0.5*pow((t[i] - t[j])/eta2_1, 2) 
                                  -2.0*pow(sin(M_PI*(t[i] - t[j])/eta3_1)/eta4_1, 2) );

                    if(i==j)
                    {
                        if (multi_instrument)
                        {
                            jit = jitters[obsi[i]-1];
                            C_1(i, j) += sig[i]*sig[i] + jit*jit;
                        }
                        else
                        {
                            C_1(i, j) += sig[i]*sig[i] + jitter1*jitter1;
                        }
                    }
                    else
                    {
                        C_1(j, i) = C_1(i, j);
                    }
                }
            }

            break;
        }

    case qpc:
        {
            /* This implements the quasi-periodic-cosine kernel from Perger+2020 */
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double tau = t[i] - t[j];
                    C_1(i, j) = exp(-0.5*pow(tau/eta2_1, 2)) * 
                                ( eta1_1*eta1_1*exp(-2.0*pow(sin(M_PI*tau/eta3_1)/eta4_1, 2)) +
                                  eta5_1*eta5_1*cos(4*M_PI*tau/eta3_1) 
                                );

                    if(i==j)
                    {
                        if (multi_instrument)
                        {
                            jit = jitters[obsi[i]-1];
                            C_1(i, j) += sig[i]*sig[i] + jit*jit;
                        }
                        else
                        {
                            C_1(i, j) += sig[i]*sig[i] + jitter1*jitter1;
                        }
                    }
                    else
                    {
                        C_1(j, i) = C_1(i, j);
                    }
                }
            }

            break;
        }


    // case celerite:
    //     {
    //         /*
    //         This implements a celerite quasi-periodic kernel devised by Andrew Collier Cameron,
    //         which satisfies k(tau=0)=amp and k'(tau=0)=0
    //         The kernel defined in the celerite paper (Eq 56 in Foreman-Mackey et al. 2017)
    //         does not satisfy k'(tau=0)=0
    //         This new kernel has only 3 parameters, eta1, eta2, eta3
    //         corresponding to an amplitude, decay timescale and period.
    //         It approximates the standard kernel with eta4=0.5
    //         */

    //         double wbeat, wrot, amp, c, d, x, a, b, e, f, g;
    //         wbeat = 1 / eta2;
    //         wrot = 2*M_PI/ eta3;
    //         amp = eta1*eta1;
    //         c = wbeat; d = wrot; x = c/d;
    //         a = amp/2; b = amp*x/2;
    //         e = amp/8; f = amp*x/4;
    //         g = amp*(3./8. + 0.001);

    //         VectorXd a_real, c_real, 
    //                 a_comp(3),
    //                 b_comp(3),
    //                 c_comp(3),
    //                 d_comp(3);
        
    //         // a_real is empty
    //         // c_real is empty
    //         a_comp << a, e, g;
    //         b_comp << b, f, 0.0;
    //         c_comp << c, c, c;
    //         d_comp << d, 2*d, 0.0;

    //         VectorXd yvar(t.size()), tt(t.size());
    //         for (int i = 0; i < t.size(); ++i){
    //             yvar(i) = sig[i] * sig[i];
    //             tt(i) = t[i];
    //         }

    //         solver.compute(
    //             jitter1*jitter1,
    //             a_real, c_real,
    //             a_comp, b_comp, c_comp, d_comp,
    //             tt, yvar  // Note: this is the measurement _variance_
    //         );

    //         break;
    //     }

    default:
        cout << "error: `kernel` should be 'standard' or 'celerite'" << endl;
        std::abort();
        break;
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "GP build matrix: ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    cout << " ns" << "\t"; // << std::endl;
    #endif
}


/**
 * @brief Fill the GP covariance matrix.
 * 
*/
void RVFWHMmodel::calculate_C_2()
{
    // Get the data
    auto data = Data::get_instance();
    const vector<double>& t = data.get_t();
    const vector<double>& sig = data.get_sig2();
    const vector<int>& obsi = data.get_obsi();
    size_t N = data.N();
    int Ni = data.Ninstruments();
    double jit;

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    switch (kernel)
    {
    case standard:
        {
            /* This implements the "standard" quasi-periodic kernel, see R&W2006 */
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    C_2(i, j) = eta1_2*eta1_2 * \
                              exp(-0.5*pow((t[i] - t[j])/eta2_2, 2) 
                                  -2.0*pow(sin(M_PI*(t[i] - t[j])/eta3_2)/eta4_2, 2) );

                    if(i==j)
                    {
                        if (multi_instrument)
                        {
                            jit = jitters[Ni + obsi[i] - 1];
                            C_2(i, j) += sig[i] * sig[i] + jit*jit;
                        }
                        else
                        {
                            C_2(i, j) += sig[i]*sig[i] + jitter2*jitter2;
                        }
                    }
                    else
                    {
                        C_2(j, i) = C_2(i, j);
                    }
                }
            }

            break;
        }

    case qpc:
        {
            /* This implements the quasi-periodic-cosine kernel from Perger+2020 */
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double tau = t[i] - t[j];
                    C_2(i, j) = exp(-0.5*pow(tau/eta2_2, 2)) * 
                                ( eta1_2*eta1_2*exp(-2.0*pow(sin(M_PI*tau/eta3_2)/eta4_2, 2)) +
                                  eta5_2*eta5_2*cos(4*M_PI*tau/eta3_2) 
                                );

                    if(i==j)
                    {
                        if (multi_instrument)
                        {
                            jit = jitters[Ni + obsi[i] - 1];
                            C_2(i, j) += sig[i]*sig[i] + jit*jit;
                        }
                        else
                        {
                            C_2(i, j) += sig[i]*sig[i] + jitter2*jitter2;
                        }
                    }
                    else
                    {
                        C_2(j, i) = C_2(i, j);
                    }
                }
            }

            break;
        }

    // case celerite:
    //     {
    //         /*
    //         This implements a celerite quasi-periodic kernel devised by Andrew Collier Cameron,
    //         which satisfies k(tau=0)=amp and k'(tau=0)=0
    //         The kernel defined in the celerite paper (Eq 56 in Foreman-Mackey et al. 2017)
    //         does not satisfy k'(tau=0)=0
    //         This new kernel has only 3 parameters, eta1, eta2, eta3
    //         corresponding to an amplitude, decay timescale and period.
    //         It approximates the standard kernel with eta4=0.5
    //         */

    //         double wbeat, wrot, amp, c, d, x, a, b, e, f, g;
    //         wbeat = 1 / eta2;
    //         wrot = 2*M_PI/ eta3;
    //         amp = eta1*eta1;
    //         c = wbeat; d = wrot; x = c/d;
    //         a = amp/2; b = amp*x/2;
    //         e = amp/8; f = amp*x/4;
    //         g = amp*(3./8. + 0.001);

    //         VectorXd a_real, c_real, 
    //                 a_comp(3),
    //                 b_comp(3),
    //                 c_comp(3),
    //                 d_comp(3);
        
    //         // a_real is empty
    //         // c_real is empty
    //         a_comp << a, e, g;
    //         b_comp << b, f, 0.0;
    //         c_comp << c, c, c;
    //         d_comp << d, 2*d, 0.0;

    //         VectorXd yvar(t.size()), tt(t.size());
    //         for (int i = 0; i < t.size(); ++i){
    //             yvar(i) = sig[i] * sig[i];
    //             tt(i) = t[i];
    //         }

    //         solver.compute(
    //             jitter1*jitter1,
    //             a_real, c_real,
    //             a_comp, b_comp, c_comp, d_comp,
    //             tt, yvar  // Note: this is the measurement _variance_
    //         );

    //         break;
    //     }

    default:
        cout << "error: `kernel` should be 'standard' or 'celerite'" << endl;
        std::abort();
        break;
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
void RVFWHMmodel::calculate_mu()
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
        mu.assign(mu.size(), bkg);
        staleness = 0;
        if(trend)
        {
            double tmid = data.get_t_middle();
            for(size_t i=0; i<t.size(); i++)
            {
                mu[i] += slope*(t[i]-tmid) + quadr*pow(t[i]-tmid, 2) + cubic*pow(t[i]-tmid, 3);
            }
        }

        if(multi_instrument)
        {
            for(size_t j=0; j<offsets.size() / 2; j++)
            {
                for(size_t i=0; i<t.size(); i++)
                {
                    if (obsi[i] == j+1) { mu[i] += offsets[j]; }
                }
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
    double P, K, phi, ecc, omega, Tp;
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
            Tp = data.M0_epoch-(P*phi)/(2.*M_PI);
            f = kepler::true_anomaly(ti, P, ecc, Tp);
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

/**
 * @brief Calculate the FWHM model
 * 
*/
void RVFWHMmodel::calculate_mu_2()
{
    auto data = Data::get_instance();
    size_t N = data.N();
    int Ni = data.Ninstruments();
    // only really needed if multi_instrument
    auto obsi = data.get_obsi();

    mu_2.assign(mu_2.size(), bkg2);

    if(multi_instrument)
    {
        for (size_t j = offsets.size() / 2; j < offsets.size(); j++)
        {
            for (size_t i = 0; i < N; i++)
            {
                if (obsi[i] == j + 2 - Ni)
                {
                    mu_2[i] += offsets[j];
                }
            }
        }
    }

}


void RVFWHMmodel::remove_known_object()
{
    auto data = Data::get_instance();
    auto t = data.get_t();
    double f, v, ti, Tp;
    // cout << "in remove_known_obj: " << KO_P[1] << endl;
    for(int j=0; j<n_known_object; j++)
    {
        for(size_t i=0; i<t.size(); i++)
        {
            ti = t[i];
            Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
            f = kepler::true_anomaly(ti, KO_P[j], KO_e[j], Tp);
            v = KO_K[j] * (cos(f+KO_w[j]) + KO_e[j]*cos(KO_w[j]));
            mu[i] -= v;
        }
    }
}

void RVFWHMmodel::add_known_object()
{
    auto data = Data::get_instance();
    auto t = data.get_t();
    double f, v, ti, Tp;
    for(int j=0; j<n_known_object; j++)
    {
        for(size_t i=0; i<t.size(); i++)
        {
            ti = t[i];
            Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
            f = kepler::true_anomaly(ti, KO_P[j], KO_e[j], Tp);
            v = KO_K[j] * (cos(f+KO_w[j]) + KO_e[j]*cos(KO_w[j]));
            mu[i] += v;
        }
    }
}


int RVFWHMmodel::is_stable() const
{
    // Get the components
    const vector< vector<double> >& components = planets.get_components();
    if (components.size() == 0)
        return 0;
    // cout << components[0].size() << endl;
    // cout << AMD::AMD_stable(components) << endl;
    return AMD::AMD_stable(components, star_mass);
}

double RVFWHMmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    auto data = Data::get_instance();
    const vector<double>& t = data.get_t();
    const vector<int>& obsi = data.get_obsi();
    auto actind = data.get_actind();
    double logH = 0.;
    double tmid = data.get_t_middle();

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
            switch (kernel)
            {
                case standard:
                {
                    if(rng.rand() <= 0.25)
                    {
                        eta1_1_prior->perturb(eta1_1, rng);
                        eta1_2_prior->perturb(eta1_2, rng);
                    }
                    else if(rng.rand() <= 0.33330)
                    {
                        eta2_1_prior->perturb(eta2_1, rng);
                        if (share_eta2)
                            eta2_2 = eta2_1;
                        else
                            eta2_2_prior->perturb(eta2_2, rng);
                    }
                    else if(rng.rand() <= 0.5)
                    {
                        eta3_1_prior->perturb(eta3_1, rng);
                        if (share_eta3)
                            eta3_2 = eta3_1;
                        else
                            eta3_2_prior->perturb(eta3_2, rng);
                    }
                    else
                    {
                        eta4_1_prior->perturb(eta4_1, rng);
                        if (share_eta4)
                            eta4_2 = eta4_1;
                        else
                            eta4_2_prior->perturb(eta4_2, rng);
                    }

                    break;
                }

                case qpc:
                {
                    if(rng.rand() <= 0.2)
                    {
                        eta1_1_prior->perturb(eta1_1, rng);
                        eta1_2_prior->perturb(eta1_2, rng);
                    }
                    else if(rng.rand() <= 0.25)
                    {
                        eta5_1_prior->perturb(eta5_1, rng);
                        if (share_eta5)
                            eta5_2 = eta5_1;
                        else
                            eta5_2_prior->perturb(eta5_2, rng);
                    }
                    else if(rng.rand() <= 0.33330)
                    {
                        eta2_1_prior->perturb(eta2_1, rng);
                        if (share_eta2)
                            eta2_2 = eta2_1;
                        else
                            eta2_2_prior->perturb(eta2_2, rng);
                    }
                    else if(rng.rand() <= 0.5)
                    {
                        eta3_1_prior->perturb(eta3_1, rng);
                        if (share_eta3)
                            eta3_2 = eta3_1;
                        else
                            eta3_2_prior->perturb(eta3_2, rng);
                    }
                    else
                    {
                        eta4_1_prior->perturb(eta4_1, rng);
                        if (share_eta4)
                            eta4_2 = eta4_1;
                        else
                            eta4_2_prior->perturb(eta4_2, rng);
                    }

                    break;
                }


                // case celerite:
                // {
                //     if(rng.rand() <= 0.33330)
                //     {
                //         log_eta1 = log(eta1);
                //         log_eta1_prior->perturb(log_eta1, rng);
                //         eta1 = exp(log_eta1);
                //     }
                //     else if(rng.rand() <= 0.5)
                //     {
                //         // log_eta2 = log(eta2);
                //         // log_eta2_prior->perturb(log_eta2, rng);
                //         // eta2 = exp(log_eta2);
                //         eta2_prior->perturb(eta2, rng);
                //     }
                //     else
                //     {
                //         eta3_prior->perturb(eta3, rng);
                //     }
                //     break;
                // }
                default:
                    break;
            }

            calculate_C_1();
            calculate_C_2();
        }
        else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
        {
            if(multi_instrument)
            {
                for (int i = 0; i < jitters.size()/2; i++)
                    Jprior->perturb(jitters[i], rng);
                for (int i = jitters.size()/2; i < jitters.size(); i++)
                    J2prior->perturb(jitters[i], rng);
            }
            else
            {
                Jprior->perturb(jitter1, rng);
                J2prior->perturb(jitter2, rng);
            }

            calculate_C_1(); // recalculate covariance matrix
            calculate_C_2();

            if (known_object)
            {
                remove_known_object();

                for (int i=0; i<n_known_object; i++){
                    KO_Pprior[i]->perturb(KO_P[i], rng);
                    KO_Kprior[i]->perturb(KO_K[i], rng);
                    KO_eprior[i]->perturb(KO_e[i], rng);
                    KO_phiprior[i]->perturb(KO_phi[i], rng);
                    KO_wprior[i]->perturb(KO_w[i], rng);
                }

                add_known_object();
            }

        }
        else // perturb other parameters: vsys, slope, offsets
        {

            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] -= bkg;
                if(trend) {
                    mu[i] -= slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for (size_t j = 0; j < offsets.size() / 2; j++)
                    {
                        if (obsi[i] == j + 1)
                        {
                            mu[i] -= offsets[j];
                        }
                    }
                }
            }

            Vprior->perturb(bkg, rng);
            C2prior->perturb(bkg2, rng);

            // propose new instrument offsets
            if (multi_instrument){
                for (size_t j = 0; j < offsets.size() / 2; j++)
                    offsets_prior->perturb(offsets[j], rng);
                for (size_t j = offsets.size() / 2; j < offsets.size(); j++)
                    offsets2_prior->perturb(offsets[j], rng);
            }

            // propose new trend
            if(trend) {
                if (degree >= 1) slope_prior->perturb(slope, rng);
                if (degree >= 2) quadr_prior->perturb(quadr, rng);
                if (degree == 3) cubic_prior->perturb(cubic, rng);
            }

            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] += bkg;
                if(trend) {
                    mu[i] += slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size() / 2; j++){
                        if (obsi[i] == j+1) { mu[i] += offsets[j]; }
                    }
                }
            }

            calculate_mu_2();

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
                Jprior->perturb(jitter1, rng);
            }
            calculate_C_1();
            // calculate_C_2();
        }
        else // perturb other parameters: vsys, slope, offsets
        {
            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] -= bkg;
                if(trend) {
                    mu[i] -= slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] -= offsets[j]; }
                    }
                }

            }

            Vprior->perturb(bkg, rng);

            // propose new instrument offsets
            if (multi_instrument){
                for(unsigned j=0; j<offsets.size(); j++)
                    offsets_prior->perturb(offsets[j], rng);
            }

            // propose new trend
            if(trend) {
                if (degree >= 1) slope_prior->perturb(slope, rng);
                if (degree >= 2) quadr_prior->perturb(quadr, rng);
                if (degree == 3) cubic_prior->perturb(cubic, rng);
            }

            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] += bkg;
                if(trend) {
                    mu[i] += slope*(t[i]-data.get_t_middle());
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] += offsets[j]; }
                    }
                }

            }
        }

    }

    else
    {
        if(rng.rand() <= 0.75) // perturb planet parameters
        {
            logH += planets.perturb(rng);
            planets.consolidate_diff();
            calculate_mu();
        }
        else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
        {
            if(multi_instrument)
            {
                for (int i = 0; i < jitters.size() / 2; i++)
                    Jprior->perturb(jitters[i], rng);
                for (int i = jitters.size() / 2; i < jitters.size(); i++)
                    J2prior->perturb(jitters[i], rng);
            }
            else
            {
                Jprior->perturb(jitter1, rng);
                J2prior->perturb(jitter2, rng);
            }

            if (studentt)
                nu_prior->perturb(nu, rng);


            if (known_object)
            {
                remove_known_object();

                for (int i=0; i<n_known_object; i++){
                    KO_Pprior[i]->perturb(KO_P[i], rng);
                    KO_Kprior[i]->perturb(KO_K[i], rng);
                    KO_eprior[i]->perturb(KO_e[i], rng);
                    KO_phiprior[i]->perturb(KO_phi[i], rng);
                    KO_wprior[i]->perturb(KO_w[i], rng);
                }

                add_known_object();
            }
        
        }
        else
        {
            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] -= bkg;
                if(trend) {
                    mu[i] -= slope*(t[i]-tmid) + quadr*pow(t[i]-tmid, 2) + cubic*pow(t[i]-tmid, 3);
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] -= offsets[j]; }
                    }
                }

            }

            // propose new vsys
            Vprior->perturb(bkg, rng);
            C2prior->perturb(bkg2, rng);

            // propose new instrument offsets
            if (multi_instrument){
                for (unsigned j = 0; j < offsets.size() / 2; j++)
                    offsets_prior->perturb(offsets[j], rng);
                for (unsigned j = offsets.size() / 2; j < offsets.size(); j++)
                    offsets2_prior->perturb(offsets[j], rng);
            }

            // propose new trend
            if(trend) {
                if (degree >= 1) slope_prior->perturb(slope, rng);
                if (degree >= 2) quadr_prior->perturb(quadr, rng);
                if (degree == 3) cubic_prior->perturb(cubic, rng);
            }

            for(size_t i=0; i<mu.size(); i++)
            {
                mu[i] += bkg;
                if(trend) {
                    mu[i] += slope*(t[i]-tmid) + quadr*pow(t[i]-tmid, 2) + cubic*pow(t[i]-tmid, 3);
                }
                if(multi_instrument) {
                    for(size_t j=0; j<offsets.size(); j++){
                        if (obsi[i] == j+1) { mu[i] += offsets[j]; }
                    }
                }
            }

            calculate_mu_2();

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
double RVFWHMmodel::log_likelihood() const
{
    auto data = Data::get_instance();
    size_t N = data.N();
    auto y = data.get_y();
    auto y2 = data.get_y2();
    auto sig = data.get_sig();
    auto obsi = data.get_obsi();

    double logL = 0.;

    if (is_stable() != 0)
        return -std::numeric_limits<double>::infinity();


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    if(GP)
    {
        /** The following code calculates the log likelihood in the case of a GP model */
        // residual vector (observed y minus model y)
        VectorXd residual(N);
        for(size_t i=0; i<N; i++)
            residual(i) = y[i] - mu[i];

        switch (kernel)
        {
            case standard:
            case qpc:
            {
                // perform the cholesky decomposition of the covariance matrix
                Eigen::LLT<Eigen::MatrixXd> cholesky;
                cholesky = C_1.llt();

                // get the lower triangular matrix L
                MatrixXd L;
                L = cholesky.matrixL();

                double logDeterminant = 0.;
                for(size_t i=0; i<N; i++)
                    logDeterminant += 2.*log(L(i,i));

                VectorXd solution;
                solution = cholesky.solve(residual);

                // y*solution
                double exponent = 0.;
                for(size_t i=0; i<N; i++)
                    exponent += residual(i)*solution(i);

                logL = -0.5*N*log(2*M_PI) - 0.5*logDeterminant - 0.5*exponent;

                // 2nd output
                for (size_t i = 0; i < N; i++)
                    residual(i) = y2[i] - mu_2[i];

                cholesky = C_2.llt();
                L = cholesky.matrixL();
                logDeterminant = 0.;
                for (size_t i = 0; i < N; i++)
                    logDeterminant += 2.*log(L(i,i));
                solution = cholesky.solve(residual);
                exponent = 0.;
                for (size_t i = 0; i < N; i++)
                    exponent += residual(i) * solution(i);
                logL += -0.5*N*log(2*M_PI) - 0.5*logDeterminant - 0.5*exponent;
                // end 2nd output

                break;
            }

            case celerite:
            {
                logL = -0.5 * (solver.dot_solve(residual) +
                    solver.log_determinant() +
                    y.size()*log(2*M_PI)); 
    
                break;
            }
        
            default:
                break;
        }

    }
    else
    {
        if (studentt){
            // The following code calculates the log likelihood 
            // in the case of a t-Student model
            double var, jit;
            for(size_t i=0; i<N; i++)
            {
                if(multi_instrument)
                {
                    jit = jitters[obsi[i]-1];
                    var = sig[i]*sig[i] + jit*jit;
                }
                else
                    var = sig[i]*sig[i] + jitter1*jitter1;

                logL += std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu)
                        - 0.5*log(M_PI*nu) - 0.5*log(var)
                        - 0.5*(nu + 1.)*log(1. + pow(y[i] - mu[i], 2)/var/nu);
            }

        }

        else{
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
                    var = sig[i]*sig[i] + jitter1*jitter1;

                logL += - halflog2pi - 0.5*log(var)
                        - 0.5*(pow(y[i] - mu[i], 2)/var);
            }
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


void RVFWHMmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    if (multi_instrument)
    {
        for (int j = 0; j < jitters.size(); j++)
            out << jitters[j] << '\t';
    }
    else
    {
        out << jitter1 << '\t';
        out << jitter2 << '\t';
    }

    if (trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
    }
        
    if (multi_instrument)
    {
        for (int j = 0; j < offsets.size(); j++)
        {
            out << offsets[j] << '\t';
        }
    }

    auto data = Data::get_instance();

    if(GP)
    {
        if (kernel == standard || kernel == qpc){
            out << eta1_1 << '\t' << eta1_2 << '\t';
            
            out << eta2_1 << '\t';
            if (!share_eta2) out << eta2_2 << '\t';
            
            out << eta3_1 << '\t';
            if (!share_eta3) out << eta3_2 << '\t';
            
            out << eta4_1 << '\t';
            if (!share_eta4) out << eta4_2 << '\t';
        }

        if (kernel == qpc){
            out << eta5_1 << '\t';
            if (!share_eta5) out << eta5_2 << '\t';
        }
        // else
        //     out << eta1_1 << '\t' << eta2_1 << '\t' << eta3_1 << '\t';
    }
    
    if(MA)
        out << sigmaMA << '\t' << tauMA << '\t';

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
    }

    planets.print(out);

    out << ' ' << staleness << ' ';

    if (studentt)
        out << '\t' << nu << '\t';

    out << bkg2 << '\t';
    out << bkg;
}

string RVFWHMmodel::description() const
{
    string desc;
    string sep = "  ";

    if (multi_instrument)
    {
        for(int j=0; j<jitters.size(); j++)
           desc += "jitter" + std::to_string(j+1) + sep;
    }
    else
    {
        desc += "jitter1" + sep;
        desc += "jitter2" + sep;
    }

    if(trend)
    {
        if (degree >= 1) desc += "slope" + sep;
        if (degree >= 2) desc += "quadr" + sep;
        if (degree == 3) desc += "cubic" + sep;
    }


    if (multi_instrument){
        for(unsigned j=0; j<offsets.size(); j++)
            desc += "offset" + std::to_string(j+1) + sep;
    }

    auto data = Data::get_instance();

    if(GP)
    {
        if(kernel == standard || kernel == qpc){
            desc += "eta1_1" + sep + "eta1_2" + sep;

            desc += "eta2_1" + sep;
            if (!share_eta2) desc += "eta2_2" + sep;

            desc += "eta3_1" + sep;
            if (!share_eta3) desc += "eta3_2" + sep;

            desc += "eta4_1" + sep;
            if (!share_eta4) desc += "eta4_2" + sep;
        }
        if (kernel == qpc){
            desc += "eta5_1" + sep;
            if (!share_eta5) desc += "eta5_2" + sep;
        }

        // else
        //     desc += "eta1_1" + sep + "eta2_1" + sep + "eta3_1" + sep;
    }
    
    if(MA)
        desc += "sigmaMA" + sep + "tauMA";

    if(known_object) { // KO mode!
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_P" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_K" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_phi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_ecc" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_w" + std::to_string(i) + sep;
    }

    desc += "ndim" + sep + "maxNp" + sep;
    if(hyperpriors)
        desc += "muP" + sep + "wP" + sep + "muK";

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        for(int i = 0; i < maxpl; i++) desc += "P" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "K" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "phi" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "ecc" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "w" + std::to_string(i) + sep;
    }

    desc += "staleness" + sep;
    if (studentt)
        desc += "nu" + sep;
    
    desc += "vsys";

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void RVFWHMmodel::save_setup() {
    auto data = Data::get_instance();
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "RVFWHMmodel" << endl << endl;

    fout << "GP: " << GP << endl;
    if (GP){
        fout << "GP_kernel: " << _kernels[kernel] << endl;
        fout << "share_eta2: " << share_eta2 << endl;
        fout << "share_eta3: " << share_eta3 << endl;
        fout << "share_eta4: " << share_eta4 << endl;
        fout << "share_eta5: " << share_eta5 << endl;
    }
    fout << "MA: " << MA << endl;
    fout << "hyperpriors: " << hyperpriors << endl;
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "multi_instrument: " << multi_instrument << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "studentt: " << studentt << endl;
    fout << endl;

    fout << endl;

    fout << "[data]" << endl;
    fout << "file: " << data.datafile << endl;
    fout << "units: " << data.dataunits << endl;
    fout << "skip: " << data.dataskip << endl;
    fout << "multi: " << data.datamulti << endl;

    fout << "files: ";
    for (auto f: data.datafiles)
        fout << f << ",";
    fout << endl;

    fout << "M0_epoch: " << data.M0_epoch << endl;

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Vprior: " << *Vprior << endl;
    fout << "C2prior: " << *C2prior << endl;
    fout << "Jprior: " << *Jprior << endl;
    fout << "J2prior: " << *J2prior << endl;

    if (trend){
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }
    if (multi_instrument)
    {
        fout << "offsets_prior: " << *offsets_prior << endl;
        fout << "offsets2_prior: " << *offsets2_prior << endl;
    }
    if (studentt)
        fout << "nu_prior: " << *nu_prior << endl;

    if (GP){
        fout << endl << "[priors.GP]" << endl;
        fout << "eta1_1_prior: " << *eta1_1_prior << endl;
        fout << "eta1_2_prior: " << *eta1_2_prior << endl;

        fout << "eta2_1_prior: " << *eta2_1_prior << endl;
        if (!share_eta2)
            fout << "eta2_2_prior: " << *eta2_2_prior << endl;

        fout << "eta3_1_prior: " << *eta3_1_prior << endl;
        if (!share_eta3)
            fout << "eta3_2_prior: " << *eta3_2_prior << endl;

        if (kernel == standard){
            fout << "eta4_1_prior: " << *eta4_1_prior << endl;
            if (!share_eta4)
                fout << "eta4_2_prior: " << *eta4_2_prior << endl;
        }

        if (kernel == qpc){
            fout << "eta5_1_prior: " << *eta5_1_prior << endl;
            if (!share_eta5)
                fout << "eta5_2_prior: " << *eta5_2_prior << endl;
        }
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
        for(int i=0; i<n_known_object; i++){
            fout << "Pprior_" << i << ": " << *KO_Pprior[i] << endl;
            fout << "Kprior_" << i << ": " << *KO_Kprior[i] << endl;
            fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *KO_wprior[i] << endl;
        }
    }

    fout << endl;
	fout.close();
}

