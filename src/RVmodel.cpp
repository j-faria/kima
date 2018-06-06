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

extern ContinuousDistribution *Cprior; // systematic velocity, m/s
extern ContinuousDistribution *Jprior; // additional white noise, m/s

extern ContinuousDistribution *slope_prior; // m/s/day

extern ContinuousDistribution *log_eta1_prior;
extern ContinuousDistribution *log_eta2_prior;
extern ContinuousDistribution *eta3_prior;
extern ContinuousDistribution *log_eta4_prior;

// from the offsets determined by Lo Curto et al. 2015 (only FGK stars)
// mean, std = 14.641789473684208, 2.7783035258938971
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
        eta1 = exp(log_eta1_prior->generate(rng)); // m/s
        // eta1 = exp(log(1E-5) + log(1E-1)*rng.rand());
        //eta1 = sqrt(3.); // m/s

        eta2 = exp(log_eta2_prior->generate(rng)); // days
        // eta2 = exp(log(1E-6) + log(1E6)*rng.rand());
        //eta2 = 50.; //days

        eta3 = eta3_prior->generate(rng); // days
        // eta3 = 15. + 35.*rng.rand();
        //eta3 = 20.; // days

        eta4 = exp(log_eta4_prior->generate(rng));
        // exp(log(1E-5) + log(1E5)*rng.rand());
        //eta4 = 0.5;
    }

    calculate_mu();

    if(GP) calculate_C();

}

void RVmodel::calculate_C()
{

    // Get the data
    const vector<double>& t = Data::get_instance().get_t();
    const vector<double>& sig = Data::get_instance().get_sig();

    #if DOCEL
        // celerite!
        // auto begin1 = std::chrono::high_resolution_clock::now();  // start timing

        /*
        This implements the kernel in Eq (61) of Foreman-Mackey et al. (2017)
        The kernel has parameters a, b, c and P
        corresponding to an amplitude, factor, decay timescale and period.
        */

        VectorXd alpha_real(1),
                 beta_real(1),
                 alpha_complex_real(1),
                 alpha_complex_imag(1),
                 beta_complex_real(1),
                 beta_complex_imag(1);
        
        //a = eta1;
        //b = eta4;
        //P = eta3;
        //c = eta2;

        alpha_real << eta1*(1.+eta4)/(2.+eta4);
        beta_real << 1./eta2;
        alpha_complex_real << eta1/(2.+eta4);
        alpha_complex_imag << 0.;
        beta_complex_real << 1./eta2;
        beta_complex_imag << 2.*M_PI / eta3;


        VectorXd yvar(t.size()), tt(t.size());
        for (int i = 0; i < t.size(); ++i){
            yvar(i) = sig[i] * sig[i] + extra_sigma * extra_sigma;
            tt(i) = t[i];
        }

        solver.compute(
            extra_sigma,
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            tt, yvar  // Note: this is the measurement _variance_
        );


        // auto end1 = std::chrono::high_resolution_clock::now();
        // cout << "new GP: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1-begin1).count() << " ns" << std::endl;
        
    #else

        int N = Data::get_instance().get_t().size();
        // auto begin = std::chrono::high_resolution_clock::now();  // start timing

        for(size_t i=0; i<N; i++)
        {
            for(size_t j=i; j<N; j++)
            {
                //C(i, j) = eta1*eta1*exp(-0.5*pow((t[i] - t[j])/eta2, 2) );
                C(i, j) = eta1*eta1*exp(-0.5*pow((t[i] - t[j])/eta2, 2) 
                           -2.0*pow(sin(M_PI*(t[i] - t[j])/eta3)/eta4, 2) );

                if(i==j)
                    C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma; //+ eta5*t[i]*t[i];
                else
                    C(j, i) = C(i, j);
            }
        }

        // auto end = std::chrono::high_resolution_clock::now();
        // cout << "old GP: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << "\t"; // << std::endl;

    #endif
}

void RVmodel::calculate_mu()
{
    auto data = Data::get_instance();
    // Get the times from the data
    const vector<double>& t = data.get_t();

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
            //cout << "J: " << extra_sigma;
            Jprior->perturb(extra_sigma, rng);
            //cout << " --> " << extra_sigma << endl;
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


    return logH;
}


double RVmodel::log_likelihood() const
{
    double logL = 0.;
    auto data = Data::get_instance();
    int N = data.N();
    const vector<double>& y = data.get_y();
    const vector<double>& sig = data.get_sig();

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

        #if DOCEL
            logL = -0.5 * (solver.dot_solve(residual) +
                           solver.log_determinant() +
                           y.size()*log(2*M_PI)); 
        #else
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
        #endif

        // calculate C^-1*(y-mu)
        // auto begin = std::chrono::high_resolution_clock::now();  // start timing
        // auto end = std::chrono::high_resolution_clock::now();
        // cout << "solve took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns \t";// <<  std::endl;

        // auto begin1 = std::chrono::high_resolution_clock::now();  // start timing
        // auto end1 = std::chrono::high_resolution_clock::now();
        // cout << "solve took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1-begin1).count() << " ns" << std::endl;

    } 
    else
    {
        /** The following code calculates the log likelihood in the case of a t-Student model*/
        //  for(size_t i=0; i<y.size(); i++)
        //  {
        //      var = sig[i]*sig[i] + extra_sigma*extra_sigma;
        //      logL += gsl_sf_lngamma(0.5*(nu + 1.)) - gsl_sf_lngamma(0.5*nu)
        //          - 0.5*log(M_PI*nu) - 0.5*log(var)
        //          - 0.5*(nu + 1.)*log(1. + pow(y[i] - mu[i], 2)/var/nu);
        //  }

        /** The following code calculates the log likelihood in the case of a Gaussian likelihood*/
        double var;
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
        // cout << "inf!" << endl;
        logL = std::numeric_limits<double>::infinity();
        // logL = -1E300;
    }
    return logL;
}

void RVmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    out<<extra_sigma<<'\t';
    
    if(trend)
        out<<slope<<'\t'; //<<quad*1E8<<'\t';

    if (obs_after_HARPS_fibers)
        out<<fiber_offset<<'\t';

    if(GP)
        out<<eta1<<'\t'<<eta2<<'\t'<<eta3<<'\t'<<eta4<<'\t';
  
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
        desc += "eta1\teta2\teta3\teta4\t";

    desc += "ndim\tmaxNp\t";
    if(hyperpriors)
        desc += "muP\twP\tmuK\t";

    desc += "Np\t";

    if (planets.get_max_num_components()>0)
        desc += "P\tK\tphi\tecc\tchi\t";

    desc += "staleness\tvsys";

    return desc;
}


void RVmodel::save_setup() {
    // save the options of the current model in a INI file
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

	fout << "obs_after_HARPS_fibers: " << obs_after_HARPS_fibers << endl;
    fout << "GP: " << GP << endl;
    fout << "hyperpriors: " << hyperpriors << endl;
    fout << "trend: " << trend << endl;
    fout << endl;
    fout << "file: " << Data::get_instance().datafile << endl;
    fout << "units: " << Data::get_instance().dataunits << endl;
    fout << "skip: " << Data::get_instance().dataskip << endl;

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
    double E;
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
    double f = acos( (cos(E)-ecc)/( 1-ecc*cos(E) ) );
    //acos gives the principal values ie [0:PI]
    //when E goes above PI we need another condition
    if(E>M_PI)
      f=2*M_PI-f;

    return f;
}
