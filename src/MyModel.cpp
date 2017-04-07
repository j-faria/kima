#include "MyModel.h"
#include "RNG.h"
#include "Utils.h"
#include "Data.h"
//#include "MultiSite2.h"
#include <cmath>
#include <fstream>
#include <chrono>
#include <typeinfo>  //for 'typeid' to work  

//#include "HODLR_Tree.hpp"
// #include <Eigen/Core>
#include "celerite/celerite.h"


using namespace std;
using namespace Eigen;
using namespace DNest4;

// get the instance for the full dataset
//DataSet& full = DataSet::getRef("full");

#define DONEW false  
#define DOCEL true

MyModel::MyModel()
:objects(5, 0, true, MyConditionalPrior())
,mu(Data::get_instance().get_t().size())
,C(Data::get_instance().get_t().size(), Data::get_instance().get_t().size())
{
    //setupHODLR();
    // celerite::solver::BandSolver<double> solver;
}


/*void MyModel::setupHODLR()
{
    const vector<double>& t = full.get_t();
    
    kernel = new QPkernel(t);
    kernel->set_hyperpars(1., 1., 1., 1.);  // not sure if this is needed

    A = new HODLR_Tree<QPkernel>(kernel, full.N, 150);
}*/

void MyModel::from_prior(RNG& rng)
{
    objects.from_prior(rng);
    objects.consolidate_diff();
    
    double ymin, ymax;
    ymin = Data::get_instance().get_y_min();
    ymax = Data::get_instance().get_y_max();

    background = ymin + (ymax - ymin)*rng.rand();

    // centered at 1 (data in m/s)
    /*extra_sigma = exp(tan(M_PI*(0.97*rng.rand() - 0.485)));*/
    // centered at 0.001 (data in km/s)
    extra_sigma = exp(-6.908 + tan(M_PI*(0.97*rng.rand() - 0.485)));

    //eta5 = exp(tan(M_PI*(0.97*rng.rand() - 0.485)));

    // Log-uniform prior from 10^(-1) to 50 m/s
    //eta1 = exp(log(1E-1) + log(5E2)*rng.rand());
    // Log-uniform prior from 10^(-5) to 0.05 km/s
    eta1 = exp(log(1E-5) + log(1E-1)*rng.rand());


    // Log-uniform prior
    eta2 = exp(log(1E-3) + log(1E2)*rng.rand());

    // or uniform prior between 10 and 40 days
    eta3 = 20. + 10.*rng.rand();

    // Log-uniform prior from 10^(-1) to 10 (fraction of eta3)
    // Log-uniform prior from 10^(-1) to 2 (fraction of eta3)
    eta4 = exp(log(1E-5) + log(1E1)*rng.rand());
    // eta4 = rng.rand();


    calculate_mu();
    calculate_C();
}

void MyModel::calculate_C()
{
    // Get the data
    const vector<double>& t = Data::get_instance().get_t();
    const vector<double>& sig = Data::get_instance().get_sig();

    #if DONEW
        //auto begin = std::chrono::high_resolution_clock::now();  // start timing

        kernel->set_hyperpars(eta1, eta2, eta3, eta4);
        cout << eta1 << "   " << eta2 << "   " << eta3 << "   " << eta4 << "   ";
        VectorXd yvar(t.size());
        for (int i = 0; i < t.size(); ++i)
            yvar(i) = eta1*eta1 + sig[i] * sig[i] + extra_sigma * extra_sigma;

        //auto begin = std::chrono::high_resolution_clock::now();  // start timing
        A->assemble_Matrix(yvar, 1e-14, 's');
        //auto end = std::chrono::high_resolution_clock::now();
        //cout << "assembling up took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;
        A->compute_Factor();

        //auto end = std::chrono::high_resolution_clock::now();

        //ofstream timerfile;
        //timerfile.open("timings.txt", std::ios_base::app);
        //timerfile.setf(ios::fixed,ios::floatfield);
        //timerfile << t.size() << '\t' << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;

    #elif DOCEL
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
        
        a = eta1;
        b = eta4;
        P = eta3;
        c = eta2;

        alpha_real << a*(1.+b)/(2.+b);
        beta_real << 1./c;
        alpha_complex_real << a/(2.+b);
        alpha_complex_imag << 0.;
        beta_complex_real << 1./c;
        beta_complex_imag << 2.*M_PI / P;


        VectorXd yvar(t.size()), tt(t.size());
        for (int i = 0; i < t.size(); ++i){
            yvar(i) = sig[i] * sig[i] + extra_sigma * extra_sigma;
            tt(i) = t[i];
        }

        solver.compute(
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



        //ofstream timerfile;
        //timerfile.open("timings.txt", std::ios_base::app);
        //timerfile.setf(ios::fixed,ios::floatfield);
        //timerfile << t.size() << '\t' << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;

    #endif
}

void MyModel::calculate_mu()
{
    // Get the times from the data
    const vector<double>& t = Data::get_instance().get_t();

    // Update or from scratch?
    bool update = (objects.get_added().size() < objects.get_components().size()) &&
            (staleness <= 10);

    // Get the components
    const vector< vector<double> >& components = (update)?(objects.get_added()):
                (objects.get_components());

    // Zero the signal
    if(!update)
    {
        mu.assign(mu.size(), background);
        staleness = 0;
    }
    else
        staleness++;

    //auto begin = std::chrono::high_resolution_clock::now();  // start timing

    double P, K, phi, ecc, viewing_angle, f, v, ti;
    for(size_t j=0; j<components.size(); j++)
    {
        P = exp(components[j][0]);
        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        viewing_angle = components[j][4];

        for(size_t i=0; i<t.size(); i++)
        {
            ti = t[i];
            f = true_anomaly(ti, P, ecc, t[0]-(P*phi)/(2.*M_PI));
            v = K*(cos(f+viewing_angle) + ecc*cos(viewing_angle));
            mu[i] += v;
        }
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //ofstream timerfile;
    //timerfile.open("timings.txt", std::ios_base::app);
    //timerfile.setf(ios::fixed,ios::floatfield);
    //timerfile << components.size() << '\t' << ecc << '\t' << viewing_angle << '\t' << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;
    
    //cout<<Ea<<endl;
}

double MyModel::perturb(RNG& rng)
{
    double logH = 0.;

    if(rng.rand() <= 0.5)
    {
        logH += objects.perturb(rng);
        objects.consolidate_diff();
        calculate_mu();
    }
    else if(rng.rand() <= 0.5)
    {
        if(rng.rand() <= 0.25)
        {
            eta1 = log(eta1);
            eta1 += log(1E4)*rng.randh(); // range of prior support
            wrap(eta1, log(1E-5), log(1E-1)); // wrap around inside prior
            eta1 = exp(eta1);
        }
        else if(rng.rand() <= 0.33330)
        {
            eta2 = log(eta2);
            eta2 += log(1E5)*rng.randh(); // range of prior support
            wrap(eta2, log(1E-3), log(1E2)); // wrap around inside prior
            eta2 = exp(eta2);
        }
        else if(rng.rand() <= 0.5)
        {
            eta3 += 10.*rng.randh(); // range of prior support
            wrap(eta3, 20., 30.); // wrap around inside prior
        }
        else
        {
            eta4 = log(eta4);
            eta4 += log(1E6)*rng.randh(); // range of prior support
            wrap(eta4, log(1E-5), log(1E1)); // wrap around inside prior
            eta4 = exp(eta4);

            // eta4 += rng.randh();
            // wrap(eta4, 0., 1.);
        }

        calculate_C();

    }
    else if(rng.rand() <= 0.5)
    {
        // data in km/s
        extra_sigma = log(extra_sigma);
        extra_sigma = (atan(extra_sigma + 6.908)/M_PI + 0.485)/0.97;
        extra_sigma += rng.randh();
        wrap(extra_sigma, 0., 1.);
        extra_sigma = -6.908 + tan(M_PI*(0.97*extra_sigma - 0.485));
        extra_sigma = exp(extra_sigma);

        calculate_C();
    }
    else
    {
        for(size_t i=0; i<mu.size(); i++)
            mu[i] -= background;

        double ymin, ymax;
        ymin = Data::get_instance().get_y_min();
        ymax = Data::get_instance().get_y_max();

        background += (ymax - ymin)*rng.randh();
        wrap(background, ymin, ymax);

        for(size_t i=0; i<mu.size(); i++)
            mu[i] += background;
    }

    return logH;
}


double MyModel::log_likelihood() const
{
    int N = Data::get_instance().get_y().size();

    /** The following code calculates the log likelihood in the case of a GP model */

    // Get the data
    const vector<double>& y = Data::get_instance().get_y();

    //auto begin = std::chrono::high_resolution_clock::now();  // start timing

    #if DONEW
        // Set up the kernel.
        //auto begin = std::chrono::high_resolution_clock::now();  // start timing
        //QPkernel kernel;
        //kernel->set_hyperpars(eta1, eta2, eta3, eta4);
        //auto end = std::chrono::high_resolution_clock::now();
        //cout << "set kernel took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;

        // Setting things up
        //auto begin = std::chrono::high_resolution_clock::now();  // start timing
        
        //auto end = std::chrono::high_resolution_clock::now();
        //cout << "setting up took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;
        
        MatrixXd b(y.size(), 1), x;
        //VectorXd yvar(t.size());
        for (int i = 0; i < y.size(); ++i) {
            //yvar(i) = eta1*eta1 + sig[i] * sig[i] + extra_sigma * extra_sigma;
            b(i, 0) = y[i] - mu[i];
        }

        //auto begin = std::chrono::high_resolution_clock::now();  // start timing
        A->solve(b, x);
        double determinant;
        A->compute_Determinant(determinant);
        //auto end = std::chrono::high_resolution_clock::now();
        //cout << "solve and determinant took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;

        //cout << logDeterminant << "   " << determinant << endl;
        //assert (logDeterminant == determinant);
        double exponent2 = 0.;
        for(int i = 0; i < y.size(); ++i)
            exponent2 += b(i,0)*x(i);


        double logL = -0.5*y.size()*log(2*M_PI)
                        - 0.5*determinant - 0.5*exponent2;

        //cout << logL << endl;
        //cout << logL << "   " << logL2 << endl;    
        //assert (logL == logL2);


    #else
        // residual vector (observed y minus model y)
        VectorXd residual(y.size());
        for(size_t i=0; i<y.size(); i++)
            residual(i) = y[i] - mu[i];

        #if DOCEL
            // logDeterminant = solver.log_determinant();
            // VectorXd solution = solver.solve(residual);

            double logL = -0.5 * (solver.dot_solve(residual) +
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

            double logL = -0.5*y.size()*log(2*M_PI)
                            - 0.5*logDeterminant - 0.5*exponent;
        #endif

        // cout << "old GP log_det: " << logDeterminant << "\t"; // << std::endl;
        // cout << "new GP log_det: " << solver.log_determinant() << std::endl;

        // calculate C^-1*(y-mu)
        // auto begin = std::chrono::high_resolution_clock::now();  // start timing
        // auto end = std::chrono::high_resolution_clock::now();
        // cout << "solve took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns \t";// <<  std::endl;

        // auto begin1 = std::chrono::high_resolution_clock::now();  // start timing
        // auto end1 = std::chrono::high_resolution_clock::now();
        // cout << "solve took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1-begin1).count() << " ns" << std::endl;
    #endif

    if(std::isnan(logL) || std::isinf(logL))
        logL = -1E300;

    //auto end = std::chrono::high_resolution_clock::now();
    ////cout << "Likelihood took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;

    //ofstream timerfile;
    //timerfile.open("timings_logL.txt", std::ios_base::app);
    //timerfile.setf(ios::fixed,ios::floatfield);
    //timerfile << y.size() << '\t' << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns" << std::endl;


    return logL;

    /** The following code calculates the log likelihood in the case of a t-Student model without correlated noise*/
    //  for(size_t i=0; i<y.size(); i++)
    //  {
    //      var = sig[i]*sig[i] + extra_sigma*extra_sigma;
    //      logL += gsl_sf_lngamma(0.5*(nu + 1.)) - gsl_sf_lngamma(0.5*nu)
    //          - 0.5*log(M_PI*nu) - 0.5*log(var)
    //          - 0.5*(nu + 1.)*log(1. + pow(y[i] - mu[i], 2)/var/nu);
    //  }

    // return logL;
}

void MyModel::print(std::ostream& out) const
{
    // output presision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    //out<<extra_sigma<<'\t'<<eta1<<'\t'<<eta2<<'\t'<<eta3<<'\t'<<eta4<<'\t'<<eta5<<'\t';
    //out<<extra_sigma<<'\t'<<eta1<<'\t'<<eta2<<'\t';
    out<<extra_sigma<<'\t'<<eta1<<'\t'<<eta2<<'\t'<<eta3<<'\t'<<eta4<<'\t';
  
    objects.print(out); out<<' '<<staleness<<' ';
    out<<background<<' ';
}

string MyModel::description() const
{
    return string("extra_sigma   eta1   eta2   eta3   eta4   objects.print   staleness   background");
    //return string("extra_sigma  eta1    eta2    objects.print   staleness   background");
    //return string("extra_sigma  eta1    eta2    eta3    eta4    eta5    offsets    objects.print   staleness   background");
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
double MyModel::ecc_anomaly(double t, double period, double ecc, double time_peri)
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
double MyModel::keplerstart3(double e, double M)
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
double MyModel::eps3(double e, double M, double x)
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
double MyModel::true_anomaly(double t, double period, double ecc, double t_peri)
{
    double E = ecc_anomaly(t, period, ecc, t_peri);
    double f = acos( (cos(E)-ecc)/( 1-ecc*cos(E) ) );
    //acos gives the principal values ie [0:PI]
    //when E goes above PI we need another condition
    if(E>M_PI)
      f=2*M_PI-f;

    return f;
}
