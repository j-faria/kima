#include "RVFWHMmodel.h"

using namespace Eigen;
#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


/// set default priors if the user didn't change them
void RVFWHMmodel::setPriors()  // BUG: should be done by only one thread!
{
    auto minFWHM = *min_element(data.actind[0].begin(), data.actind[0].end());
    auto maxFWHM = *max_element(data.actind[0].begin(), data.actind[0].end());
    auto spanFWHM = maxFWHM - minFWHM;

    // systemic velocity
    if (!Cprior)
        Cprior = make_prior<Uniform>(data.get_RV_min(), data.get_RV_max());
    
    // "systemic FWHM"
    if (!C2prior)
        C2prior = make_prior<Uniform>(minFWHM, maxFWHM);

    // jitter for the RVs
    if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(
            min(1.0, 0.1 * data.get_max_RV_span()), data.get_max_RV_span());

    // jitter for the FWHM
    if (!J2prior)
        J2prior = make_prior<ModifiedLogUniform>(min(1.0, 0.1 * spanFWHM), spanFWHM);

    if (trend){
        if (degree == 0)
            throw std::logic_error("trend=true but degree=0");
        if (degree > 3)
            throw std::range_error("can't go higher than 3rd degree trends");
        if (degree >= 1 && !slope_prior)
            slope_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(1)) );
        if (degree >= 2 && !quadr_prior)
            quadr_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(2)) );
        if (degree == 3 && !cubic_prior)
            cubic_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(3)) );
    }

    // if offsets_prior is not (re)defined, assume a default
    if (data.datamulti)
    {
        if (!offsets_prior)
            offsets_prior = make_prior<Uniform>( -data.get_RV_span(), data.get_RV_span() );
        if (!offsets_fwhm_prior) {
            offsets_fwhm_prior = make_prior<Uniform>( -spanFWHM, spanFWHM );
        }

        for (size_t j = 0; j < data.number_instruments - 1; j++)
        {
            // if individual_offset_prior is not (re)defined, assume offsets_prior
            if (!individual_offset_prior[j])
                individual_offset_prior[j] = offsets_prior;
            if (!individual_offset_fwhm_prior[j])
                individual_offset_fwhm_prior[j] = offsets_fwhm_prior;
        }
    }

    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior)");
        }
    }

    /* GP parameters */
    if (!eta1_prior)
        eta1_prior = make_prior<LogUniform>(0.1, 100);
    if (!eta1_fwhm_prior)
        eta1_fwhm_prior = make_prior<LogUniform>(0.1, spanFWHM);

    if (!eta2_prior)
        eta2_prior = make_prior<LogUniform>(1, 100);
    if (!eta2_fwhm_prior && !share_eta2)
        eta2_fwhm_prior = make_prior<LogUniform>(1, 100);

    if (!eta3_prior)
        eta3_prior = make_prior<Uniform>(10, 40);
    if (!eta3_fwhm_prior && !share_eta3)
        eta3_fwhm_prior = make_prior<Uniform>(10, 40);

    if (!eta4_prior)
        eta4_prior = make_prior<Uniform>(0.2, 5);
    if (!eta4_fwhm_prior && !share_eta4)
        eta4_fwhm_prior = make_prior<Uniform>(0.2, 5);

    if (!alpha_prior && kernel == perrq)
        alpha_prior = make_prior<LogUniform>(0.5, 10);

    if (!eta5_prior && kernel == qpc)
        eta5_prior = make_prior<Uniform>(0, 10);

}


void RVFWHMmodel::from_prior(RNG& rng)
{
    if (data.actind.size() < 2) {
        std::cout << std::endl;
        throw std::runtime_error("no activity indicators were loaded");
    }

    // preliminaries
    setPriors();
    save_setup();


    planets.from_prior(rng);
    planets.consolidate_diff();

    bkg = Cprior->generate(rng);
    bkg_fwhm = C2prior->generate(rng);

    if(data.datamulti)
    {
        // draw instrument offsets for the RVs
        for (size_t i = 0; i < offsets.size() / 2; i++)
            offsets[i] = offsets_prior->generate(rng);
        // draw instrument offsets for the FWHM
        for (size_t i = offsets.size() / 2; i < offsets.size(); i++)
            offsets[i] = offsets_fwhm_prior->generate(rng);

        for (size_t i = 0; i < jitters.size() / 2; i++)
            jitters[i] = Jprior->generate(rng);
        for (size_t i = jitters.size() / 2; i < jitters.size(); i++)
            jitters[i] = J2prior->generate(rng);
    }
    else
    {
        jitter = Jprior->generate(rng);
        jitter_fwhm = J2prior->generate(rng);
    }

    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }


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

    // GP
    eta1 = eta1_prior->generate(rng);  // m/s
    eta1_fw = eta1_fwhm_prior->generate(rng);  // m/s

    if (kernel == sqexp)
    {
        eta2 = eta2_prior->generate(rng); // days
        if (!share_eta2)
            eta2_fw = eta2_fwhm_prior->generate(rng); // days
    }
    else
    {
        eta3 = eta3_prior->generate(rng); // days
        if (!share_eta3)
            eta3_fw = eta3_fwhm_prior->generate(rng); // days

        eta2 = eta2_prior->generate(rng); // days
        if (!share_eta2)
            eta2_fw = eta2_fwhm_prior->generate(rng); // days

        eta4 = exp(eta4_prior->generate(rng));
        if (!share_eta4)
            eta4_fw = eta4_fwhm_prior->generate(rng);
        
        if (kernel == perrq)
            alpha = alpha_prior->generate(rng);
            //! missing FWHM

        if (kernel == qpc)
            eta5 = eta5_prior->generate(rng);
            //! missing FWHM
    }

    calculate_mu();
    calculate_mu_fwhm();

    calculate_C();
    calculate_C_fwhm();
}

/// @brief Calculate the full RV model
void RVFWHMmodel::calculate_mu()
{
    size_t N = data.N();

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
            for(size_t i=0; i<N; i++)
            {
                mu[i] += slope * (data.t[i] - tmid) +
                         quadr * pow(data.t[i] - tmid, 2) +
                         cubic * pow(data.t[i] - tmid, 3);
            }
        }

        if(data.datamulti)
        {
            for (size_t j = 0; j < offsets.size() / 2; j++)
            {
                for (size_t i = 0; i < N; i++)
                {
                    if (data.obsi[i] == j+1) { mu[i] += offsets[j]; }
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
        if(false) //hyperpriors
            P = exp(components[j][0]);
        else
            P = components[j][0];

        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];

        auto v = brandt::keplerian(data.t, P, K, ecc, omega, phi, data.M0_epoch);
        for(size_t i=0; i<N; i++)
            mu[i] += v[i];
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


/// @brief Calculate the full FWHM model
void RVFWHMmodel::calculate_mu_fwhm()
{
    size_t N = data.N();
    int Ni = data.Ninstruments();

    mu_fwhm.assign(mu_fwhm.size(), bkg_fwhm);

    if (data.datamulti) {
        auto obsi = data.get_obsi();
        for (size_t j = offsets.size() / 2; j < offsets.size(); j++) {
            for (size_t i = 0; i < N; i++) {
                if (obsi[i] == j + 2 - Ni) {
                    mu_fwhm[i] += offsets[j];
                }
            }
        }
    }
}


/// @brief Fill the GP covariance matrix
void RVFWHMmodel::calculate_C()
{
    size_t N = data.N();

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    switch (kernel)
    {
    case standard:
        {
            /* This implements the "standard" quasi-periodic kernel, see R&W2006 */
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = i; j < N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    C(i, j) = eta1*eta1*exp(-0.5*pow(r/eta2, 2)
                                -2.0*pow(sin(M_PI*r/eta3)/eta4, 2) );

                    if (i == j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i] - 1];
                            C(i, j) += sig * sig + jit * jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter * jitter;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    case permatern32:
        {
            // This implements a quasi-periodic kernel built from the 
            // Matern 3/2 kernel, see R&W2006
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    double s = 2 * abs(sin(M_PI * r / eta3));

                    C(i, j) = eta1 * eta1 * exp(-0.5*pow(r/eta2, 2)) * (1 + sqrt(3)*s/eta4) * exp(-sqrt(3)*s/eta4);

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig*sig + jit*jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter * jitter;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    case permatern52:
        {
            // This implements a quasi-periodic kernel built from the 
            // Matern 5/2 kernel, see R&W2006
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    double s = 2 * abs(sin(M_PI * r / eta3));

                    C(i, j) = eta1 * eta1 \
                              * exp(-0.5*pow(r/eta2, 2)) \
                              * ( 1 + sqrt(5)*s/eta4 + 5*s*s/(3*eta4*eta4) ) * exp(-sqrt(5)*s/eta4);

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig*sig + jit*jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter * jitter;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    case perrq:
        {
            // This implements a quasi-periodic kernel built from the 
            // Rational Quadratic kernel, see R&W2006
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    double s = abs(sin(M_PI * r / eta3));

                    C(i, j) = eta1 * eta1 \
                              * exp(-0.5*pow(r/eta2, 2)) \
                              * pow(1 + 2*s*s/(alpha*eta4*eta4), -alpha);

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig*sig + jit*jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter * jitter;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    case sqexp:
        {
            /* This implements the squared exponential kernel, see R&W2006 */
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    C(i, j) = eta1 * eta1 * exp(-0.5 * pow(r / eta2, 2));

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig*sig + jit*jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter * jitter;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    /* This implements the quasi-periodic-cosine kernel from Perger+2020 */
    case qpc:
        {
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    C(i, j) = exp(-0.5*pow(r/eta2, 2)) * 
                                (eta1*eta1*exp(-2.0*pow(sin(M_PI*r/eta3)/eta4, 2)) + eta5*eta5*cos(4*M_PI*r/eta3) );

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig*sig + jit*jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter * jitter;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    /* This implements the periodic (or locally periodic) kernel, see R&W2006 */
    case periodic:
        {
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    C(i, j) = eta1*eta1*exp(-2.0*pow(sin(M_PI*r/eta3)/eta4, 2) );

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig*sig + jit*jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter * jitter;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    default:
        cout << "error: `kernel` should be one of" << endl;
        cout << "'standard', ";
        cout << "'qpc', ";
        cout << "'permatern32', ";
        cout << "'permatern52', ";
        cout << "'perrq', ";
        cout << "'periodic', ";
        cout << "or 'sqexp'" << endl;
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


/// @brief Fill the GP covariance matrix
void RVFWHMmodel::calculate_C_fwhm()
{
    size_t N = data.N();
    auto t = data.get_t();
    auto sig = data.get_actind()[1];

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    switch (kernel)
    {
    case standard:
        {
            /* This implements the "standard" quasi-periodic kernel, see R&W2006 */
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = i; j < N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    C_fwhm(i, j) = eta1_fw * eta1_fw * exp(-0.5 * pow(r / eta2_fw, 2) - 2.0 * pow(sin(M_PI * r / eta3_fw) / eta4_fw, 2));

                    if (i == j)
                    {
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i] - 1];
                            C_fwhm(i, j) += sig[i] * sig[i] + jit * jit;
                        }
                        else
                        {
                            C_fwhm(i, j) += sig[i] * sig[i] + jitter * jitter;
                        }
                    }
                    else
                    {
                        C_fwhm(j, i) = C_fwhm(i, j);
                    }
                }
            }

            break;
        }

    case permatern32:
        {
            // This implements a quasi-periodic kernel built from the 
            // Matern 3/2 kernel, see R&W2006
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    double s = 2 * abs(sin(M_PI * r / eta3_fw));

                    C(i, j) = eta1_fw * eta1_fw * exp(-0.5*pow(r/eta2_fw, 2)) * (1 + sqrt(3)*s/eta4_fw) * exp(-sqrt(3)*s/eta4_fw);

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig*sig + jit*jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter_fwhm * jitter_fwhm;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    case permatern52:
        {
            // This implements a quasi-periodic kernel built from the 
            // Matern 5/2 kernel, see R&W2006
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    double s = 2 * abs(sin(M_PI * r / eta3_fw));

                    C(i, j) = eta1_fw * eta1_fw * exp(-0.5 * pow(r / eta2_fw, 2)) * (1 + sqrt(5) * s / eta4_fw + 5 * s * s / (3 * eta4_fw * eta4_fw)) * exp(-sqrt(5) * s / eta4_fw);

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig * sig + jit * jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter_fwhm * jitter_fwhm;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    case perrq:
        {
            // This implements a quasi-periodic kernel built from the 
            // Rational Quadratic kernel, see R&W2006
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    double s = abs(sin(M_PI * r / eta3_fw));

                    C(i, j) = eta1_fw * eta1_fw * exp(-0.5 * pow(r / eta2_fw, 2)) * pow(1 + 2 * s * s / (alpha * eta4_fw * eta4_fw), -alpha);

                    if (i == j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig*sig + jit*jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter_fwhm * jitter_fwhm;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    case sqexp:
        {
            /* This implements the squared exponential kernel, see R&W2006 */
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = i; j < N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    C(i, j) = eta1_fw * eta1_fw * exp(-0.5 * pow(r / eta2_fw, 2));

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i]-1];
                            C(i, j) += sig * sig + jit * jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter_fwhm * jitter_fwhm;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    /* This implements the quasi-periodic-cosine kernel from Perger+2020 */
    case qpc:
        {
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    C(i, j) = exp(-0.5 * pow(r / eta2_fw, 2)) *
                              (eta1_fw * eta1_fw * exp(-2.0 * pow(sin(M_PI * r / eta3_fw) / eta4_fw, 2)) + eta5_fw * eta5_fw * cos(4 * M_PI * r / eta3_fw));

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i] - 1];
                            C(i, j) += sig * sig + jit * jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter_fwhm * jitter_fwhm;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    /* This implements the periodic (or locally periodic) kernel, see R&W2006 */
    case periodic:
        {
            for(size_t i=0; i<N; i++)
            {
                for(size_t j=i; j<N; j++)
                {
                    double r = data.t[i] - data.t[j];
                    C(i, j) = eta1_fw * eta1_fw * exp(-2.0 * pow(sin(M_PI * r / eta3_fw) / eta4_fw, 2));

                    if(i==j)
                    {
                        double sig = data.sig[i];
                        if (data.datamulti)
                        {
                            double jit = jitters[data.obsi[i] - 1];
                            C(i, j) += sig * sig + jit * jit;
                        }
                        else
                        {
                            C(i, j) += sig * sig + jitter_fwhm * jitter_fwhm;
                        }
                    }
                    else
                    {
                        C(j, i) = C(i, j);
                    }
                }
            }

            break;
        }

    default:
        cout << "error: `kernel` should be one of" << endl;
        cout << "'standard', ";
        cout << "'qpc', ";
        cout << "'permatern32', ";
        cout << "'permatern52', ";
        cout << "'perrq', ";
        cout << "'periodic', ";
        cout << "or 'sqexp'" << endl;
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


void RVFWHMmodel::remove_known_object()
{
    double f, v, ti, Tp;
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] -= v[i];
        }
    }
}

void RVFWHMmodel::add_known_object()
{
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += v[i];
        }
    }
}


int RVFWHMmodel::is_stable() const
{
    // Get the components
    const vector< vector<double> >& components = planets.get_components();
    if (components.size() == 0 && !known_object)
        return 0;
    
    int stable_planets = 0;
    int stable_known_object = 0;

    if (components.size() != 0)
        stable_planets = AMD::AMD_stable(components, star_mass);

    if (known_object) {
        vector<vector<double>> ko_components;
        ko_components.resize(n_known_object);
        for (int j = 0; j < n_known_object; j++) {
            ko_components[j] = {KO_P[j], KO_K[j], KO_phi[j], KO_e[j], KO_w[j]};
        }
        
        stable_known_object = AMD::AMD_stable(ko_components, star_mass);
    }

    return stable_planets + stable_known_object;
}


double RVFWHMmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    auto actind = data.get_actind();
    double logH = 0.;
    double tmid = data.get_t_middle();


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
            case qpc:
            case permatern32:
            case permatern52:
            case perrq:
            case periodic:
            {
                if(rng.rand() <= 0.25)
                {
                    eta1_prior->perturb(eta1, rng);
                    eta1_fwhm_prior->perturb(eta1_fw, rng);

                    if (kernel == qpc) {
                        eta5_prior->perturb(eta5, rng);
                        eta5_fwhm_prior->perturb(eta5_fw, rng);
                    }
                }
                else if(rng.rand() <= 0.33330)
                {
                    eta3_prior->perturb(eta3, rng);
                    if (share_eta3)
                        eta3_fw = eta3;
                    else
                        eta3_fwhm_prior->perturb(eta3_fw, rng);
                }
                else if(rng.rand() <= 0.5)
                {
                    eta2_prior->perturb(eta2, rng);
                    if (share_eta2)
                        eta2_fw = eta2;
                    else
                        eta2_fwhm_prior->perturb(eta2_fw, rng);
                }
                else
                {
                    eta4_prior->perturb(eta4, rng);
                    if (share_eta4)
                        eta4_fw = eta4;
                    else
                        eta4_fwhm_prior->perturb(eta4_fw, rng);

                    if (kernel == perrq)
                        alpha_prior->perturb(alpha, rng);
                }

                break;
            }

            case sqexp:
            {
                if(rng.rand() <= 0.5)
                {
                    eta1_prior->perturb(eta1, rng);
                    eta1_fwhm_prior->perturb(eta1_fw, rng);
                }
                else
                {
                    eta2_prior->perturb(eta2, rng);
                    if (share_eta2)
                        eta2_fw = eta2;
                    else
                        eta2_fwhm_prior->perturb(eta2_fw, rng);
                }
            }
            
            default:
                break;
        }

        calculate_C();
        calculate_C_fwhm();
    }

    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        if(data.datamulti)
        {
            for(int i=0; i<jitters.size(); i++)
                Jprior->perturb(jitters[i], rng);
        }
        else
        {
            Jprior->perturb(jitter, rng);
            J2prior->perturb(jitter_fwhm, rng);
        }

        calculate_C(); // recalculate covariance matrix
        calculate_C_fwhm(); // recalculate covariance matrix

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
                mu[i] -= slope * (data.t[i] - tmid) +
                            quadr * pow(data.t[i] - tmid, 2) +
                            cubic * pow(data.t[i] - tmid, 3);
            }
            if(data.datamulti) {
                for(size_t j=0; j<offsets.size(); j++){
                    if (data.obsi[i] == j+1) { mu[i] -= offsets[j]; }
                }
            }

        }

        // propose new vsys
        Cprior->perturb(bkg, rng);
        C2prior->perturb(bkg_fwhm, rng);

        // propose new instrument offsets
        if (data.datamulti){
            for (size_t j = 0; j < offsets.size() / 2; j++)
                offsets_prior->perturb(offsets[j], rng);
            for (size_t j = offsets.size() / 2; j < offsets.size(); j++)
                offsets_fwhm_prior->perturb(offsets[j], rng);
        }

        // propose new slope
        if(trend) {
            if (degree >= 1) slope_prior->perturb(slope, rng);
            if (degree >= 2) quadr_prior->perturb(quadr, rng);
            if (degree == 3) cubic_prior->perturb(cubic, rng);
        }


        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += bkg;
            if(trend) {
                mu[i] += slope * (data.t[i] - tmid) +
                            quadr * pow(data.t[i] - tmid, 2) +
                            cubic * pow(data.t[i] - tmid, 3);
            }
            if(data.datamulti) {
                for(size_t j=0; j<offsets.size(); j++){
                    if (data.obsi[i] == j+1) { mu[i] += offsets[j]; }
                }
            }
        }

        calculate_mu_fwhm();

    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Perturb took ";
    cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    cout << " μs" << std::endl;
    #endif

    return logH;
}


double RVFWHMmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& y = data.get_y();
    const auto& sig = data.get_sig();
    const auto& obsi = data.get_obsi();

    double logL = 0.;

    if (enforce_stability){
        int stable = is_stable();
        if (stable != 0)
            return -std::numeric_limits<double>::infinity();
    }


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    // residual vector (observed y minus model y)
    VectorXd residual(y.size());
    for (size_t i = 0; i < y.size(); i++)
        residual(i) = y[i] - mu[i];

    // perform the cholesky decomposition of the covariance matrix
    Eigen::LLT<Eigen::MatrixXd> cholesky;
    cholesky = C.llt();
    // get the lower triangular matrix L
    MatrixXd L = cholesky.matrixL();
    double logDeterminant = 0.;
    for (size_t i = 0; i < N; i++)
        logDeterminant += 2. * log(L(i, i));

    VectorXd solution = cholesky.solve(residual);
    // y*solution
    double exponent = 0.;
    for (size_t i = 0; i < N; i++)
        exponent += residual(i) * solution(i);

    logL += -0.5 * N * log(2 * M_PI) - 0.5 * logDeterminant - 0.5 * exponent;

    // 2nd output
    for (size_t i = 0; i < N; i++)
        residual(i) = data.actind[0][i] - mu_fwhm[i];

    cholesky = C_fwhm.llt();
    L = cholesky.matrixL();
    logDeterminant = 0.;
    for (size_t i = 0; i < N; i++)
        logDeterminant += 2. * log(L(i, i));
    solution = cholesky.solve(residual);
    exponent = 0.;
    for (size_t i = 0; i < N; i++)
        exponent += residual(i) * solution(i);
    logL += -0.5 * N * log(2 * M_PI) - 0.5 * logDeterminant - 0.5 * exponent;


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

    if (data.datamulti)
    {
        for (int j = 0; j < jitters.size(); j++)
            out << jitters[j] << '\t';
    }
    else {
        out << jitter << '\t';
        out << jitter_fwhm << '\t';
    }

    if(trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
    }
        
    if (data.datamulti){
        for (int j = 0; j < offsets.size(); j++)
        {
            out << offsets[j] << '\t';
        }
    }

    // write GP parameters
    out << eta1 << '\t' << eta1_fw << '\t';
    
    out << eta2 << '\t';
    if (!share_eta2) out << eta2_fw << '\t';
    
    out << eta3 << '\t';
    if (!share_eta3) out << eta3_fw << '\t';
    
    out << eta4 << '\t';
    if (!share_eta4) out << eta4_fw << '\t';

    if (kernel == qpc){
        out << eta5 << '\t';
        if (!share_eta5) out << eta5_fw << '\t';
    }


    // write KO parameters
    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
    }

    // write planet parameters
    planets.print(out);

    out << staleness << '\t';

    out << bkg_fwhm << '\t';
    out << bkg;
}


string RVFWHMmodel::description() const
{
    string desc;
    string sep = "   ";

    if (data.datamulti)
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


    if (data.datamulti){
        for(unsigned j=0; j<offsets.size(); j++)
            desc += "offset" + std::to_string(j+1) + sep;
    }

    // GP parameters
    desc += "eta1" + sep + "eta1_fw" + sep;

    desc += "eta2" + sep;
    if (!share_eta2) desc += "eta2_fw" + sep;

    desc += "eta3" + sep;
    if (!share_eta3) desc += "eta3_fw" + sep;

    desc += "eta4" + sep;
    if (!share_eta4) desc += "eta4_fw" + sep;

    if (kernel == qpc){
        desc += "eta5" + sep;
        if (!share_eta5) desc += "eta5_fw" + sep;
    }


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
    if(false) // hyperpriors
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

    desc += "C2" + sep;
    desc += "vsys";

    return desc;
}


void RVFWHMmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "RVFWHMmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "GP: " << true << endl;
    fout << "GP_kernel: " << _kernels[kernel] << endl;
    fout << "share_eta2: " << share_eta2 << endl;
    fout << "share_eta3: " << share_eta3 << endl;
    fout << "share_eta4: " << share_eta4 << endl;
    fout << "share_eta5: " << share_eta5 << endl;
    fout << "share_eta6: " << share_eta6 << endl;

    fout << "hyperpriors: " << false << endl;
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "multi_instrument: " << data.datamulti << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
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

    fout.precision(15);
    fout << "M0_epoch: " << data.M0_epoch << endl;
    fout.precision(6);

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Cprior: " << *Cprior << endl;
    fout << "Jprior: " << *Jprior << endl;
    if (trend){
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }
    if (data.datamulti)
        fout << "offsets_prior: " << *offsets_prior << endl;

    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        if (false){
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

