#include "kepler.h"

const double TWO_PI = M_PI * 2;

// modulo 2pi
double mod2pi(const double &angle) {
    if (angle < TWO_PI && angle >= 0) return angle;

    if (angle >= TWO_PI)
    {
        const double M = angle - TWO_PI;
        if (M > TWO_PI)
            return fmod(M, TWO_PI);
        else
            return M;
    }
    else {
        const double M = angle + TWO_PI;
        if (M < 0)
            return fmod(M, TWO_PI) + TWO_PI;
        else
            return M;
    }
}


// A solver for Kepler's equation based on
// "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006
namespace murison
{

    double solver(double M, double ecc)
    {
        double tol;
        if (ecc < 0.8)
            tol = 1e-14;
        else
            tol = 1e-13;

        double Mnorm = mod2pi(M);
        double E0 = start3(ecc, Mnorm);
        double dE = tol + 1;
        double E = M;
        int count = 0;
        while (dE > tol)
        {
            E = E0 - eps3(ecc, Mnorm, E0);
            dE = std::abs(E - E0);
            E0 = E;
            count++;
            // failed to converge, this only happens for nearly parabolic orbits
            if (count == 100)
                break;
        }
        return E;
    }

    std::vector<double> solver(const std::vector<double> &M, double ecc)
    {
        std::vector<double> E(M.size());
        for (size_t i = 0; i < M.size(); i++)
            E[i] = solver(M[i], ecc);
        return E;
    }

    /**
        Calculates the eccentric anomaly at time t by solving Kepler's equation.

        @param t the time at which to calculate the eccentric anomaly.
        @param period the orbital period of the planet
        @param ecc the eccentricity of the orbit
        @param t_peri time of periastron passage
        @return eccentric anomaly.
    */
    double ecc_anomaly(double t, double period, double ecc, double time_peri)
    {
        double n = 2. * M_PI / period;  // mean motion
        double M = n * (t - time_peri); // mean anomaly
        return solver(M, ecc);
    }

    /**
        Provides a starting value to solve Kepler's equation.
        See "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006

        @param e the eccentricity of the orbit
        @param M mean anomaly (in radians)
        @return starting value for the eccentric anomaly.
    */
    double start3(double e, double M)
    {
        double t34 = e * e;
        double t35 = e * t34;
        double t33 = cos(M);
        return M + (-0.5 * t35 + e + (t34 + 1.5 * t33 * t35) * t33) * sin(M);
    }

    /**
        An iteration (correction) method to solve Kepler's equation.
        See "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006

        @param e the eccentricity of the orbit
        @param M mean anomaly (in radians)
        @param x starting value for the eccentric anomaly
        @return corrected value for the eccentric anomaly
    */
    double eps3(double e, double M, double x)
    {
        double t1 = cos(x);
        double t2 = -1 + e * t1;
        double t3 = sin(x);
        double t4 = e * t3;
        double t5 = -x + t4 + M;
        double t6 = t5 / (0.5 * t5 * t4 / t2 + t2);

        return t5 / ((0.5 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2);
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
    double true_anomaly(double t, double period, double ecc, double t_peri)
    {
        double E = ecc_anomaly(t, period, ecc, t_peri);
        double cosE = cos(E);
        double f = acos((cosE - ecc) / (1 - ecc * cosE));
        // acos gives the principal values ie [0:PI]
        // when E goes above PI we need another condition
        if (E > M_PI)
            f = TWO_PI - f;
        return f;
    }


    //
    std::vector<double> keplerian(const std::vector<double> &t, double P,
                                  double K, double ecc, double w, double M0,
                                  double M0_epoch) 
    {
      // allocate RVs
      std::vector<double> rv(t.size());

      // mean motion, once per orbit
      double n = 2. * M_PI / P;
      // sin and cos of argument of periastron, once per orbit
      double sinw, cosw;
      __sincos(w, &sinw, &cosw);
      // ecentricity factor for g, once per orbit
      double g_e = sqrt((1 + ecc) / (1 - ecc));

      for (size_t i = 0; i < t.size(); i++) {
        double E, cosE;
        double M = n * (t[i] - M0_epoch) - M0;
        E = solver(M, ecc);
        // sincos(E, &sinE, &cosE);
        cosE = cos(E);
        double f = acos((cosE - ecc) / (1 - ecc * cosE));
        // acos gives the principal values ie [0:PI]
        // when E goes above PI we need another condition
        if (E > M_PI)
          f = TWO_PI - f;
        rv[i] = K * (cos(f + w) + ecc * cosw);
      }

      return rv;
    }

} // namespace murison


namespace postKep
{
    double c_light = 299792458; //m/s
    double period_correction(double p_obs, double wdot)
    {
        //convert observational period (in days) to anomalistic period (in days) using wdot (in arcsec per year)
        double P_anom = p_obs*(1+wdot*0.0000000132733744*p_obs/TWO_PI);
    
        return P_anom;
    }
    
    double change_omega(double w, double wdot, double ti, double Tp)
    {
        //linear change in omega and convert from arcsec/year to radians/day
        double dw = wdot*(ti-Tp)*0.0000000132733744;
    
        return w + dw;
    }
    
    inline double semiamp(double M0, double M1, double P, double ecc)
    {
        // given M0 and M1 in solar mass and P in days, returns in m/s
        double m1 = M1 * 1047.5655; //in jupiter mass
        double m01 = (M0 + M1); //in solar mass
        
        return 28.4329*pow((1-pow(ecc,2.0)),-0.5)*m1*pow(m01,-2.0/3)*pow(P/365,-1.0/3);
    }
    
    inline double f_M(double K,double M0, double M1, double P, double ecc)
    {
        double f = semiamp(M0, M1, P, ecc) - K;
    
        return f;
    }

    inline double f_dash_M(double K,double M0, double M1, double P, double ecc)
    {
        // given M0 and M1 in solar mass and P in days, returns in m/s
        double MjMs = 1047.5655;
        double m1 = M1 * MjMs; //in jupiter mass
        double m01 = (M0 + M1); //in solar mass
    
        double f_dash_1 = 28.4329*pow((1-pow(ecc,2.0)),-0.5)*MjMs*pow(m01,-2.0/3)*pow(P/365,-1.0/3);
        double f_dash_2 = -2*28.4329*pow((1-pow(ecc,2.0)),-0.5)*m1*pow(m01,-5.0/3)*pow(P/365,-1.0/3)/3;
    
        return f_dash_1 - f_dash_2;
    }

    inline double get_K2_v2(double K1, double M, double P, double ecc)
    {
        double M_est = (K1/28.4329)*pow((1-pow(ecc,2.0)),0.5)*pow(M,2.0/3)*pow((P/365),1.0/3)/1047.5655;
        double k = semiamp(M, M_est, P, ecc);
        while(abs(k-K1)>50)
        {
            M_est = M_est - f_M(K1, M, M_est, P, ecc)/f_dash_M(K1, M, M_est, P, ecc);
            k = semiamp(M, M_est, P, ecc);
        }
        double K2 = K1*M/M_est;
    
        return K2;

    }

    inline double get_K2_v1(double K1, double M, double P, double ecc)
    {
        double M_est = (K1/28.4329)*pow((1-pow(ecc,2.0)),0.5)*pow(M,2.0/3)*pow((P/365),1.0/3)/1047.5655;
        double a = M_est/3;
        double b = M_est*3;
        double c = (a+b)/2;
        double eps = semiamp(M,b,P,ecc) - semiamp(M,a,P,ecc);
    
        while (abs(eps) > 50) {
            c = (b+a)/2;
            double x = K1 - semiamp(M,c,P,ecc);
            if (x < 0) {
                b = c;
            } else {
                a = c;
            }
            eps = semiamp(M,b,P,ecc) - semiamp(M,a,P,ecc);
        }
        double M2 = c;
        double K2 =K1*M/M_est;
        
        return K2;
    }
    
    inline double light_travel_time(double K1, double f, double w, double ecc)
    {
        double delta_LT = pow(K1,2.0)*pow(sin(f + w), 2.0)*(1+ecc*cos(f))/c_light;
    
        return delta_LT;
    }
    
    inline double transverse_doppler(double K1, double f, double ecc)
    {
        //here assume inclination of 90 (eclipsing) so sin(i) = 1
        double sini = 1.0;
        double delta_TD = pow(K1,2.0)*(1 + ecc*cos(f) - (1-pow(ecc,2.0))/2)/(c_light*pow(sini,2.0));
    
        return delta_TD;
    }
    
    inline double gravitational_redshift(double K1, double K2, double f, double ecc)
    {
        //again assume inclination of 90 (eclipsing) so sin(i) = 1
        double sini = 1.0;
        double delta_GR = K1*(K1+K2)*(1+ecc*cos(f))/(c_light*pow(sini,2.0));
    
        return delta_GR;
    }
    
    inline double v_tide(double R1, double M1, double M2, double P, double f, double w)
    {
        double phi_0 = M_PI/2 - w;
        double MjMs = 1047.5655;
        M2 = M2*MjMs;
        
        return 1.13*M2/(M1*(M1+M2/MjMs))*pow(R1,4.0)*pow(P,-3.0)*sin(2*(f-phi_0));
    }
    
    double post_Newtonian(double K1, double f, double ecc, double w, double P, double M1, double M2, double R1)
    {
        double K2;
        double v = 0.0;
        //if M2 is specified use it, if not, numerically calculate it
        if (relativistic_correction || tidal_correction)
        {
            if (M2 > 0.01)
            {
                K2 = K1*M1/M2;
            }
            else
            {
                K2 = get_K2_v2(K1, M1, P, ecc); 
                M2 = K1*M1/K2;
            }
        }
        if (tidal_correction)
        {
            if (R1 <0.01)
            {
                R1 = pow(M1,0.8);
            }
            double delta_v_tide = v_tide(R1,M1,M2,P,f,w);
            v = v + delta_v_tide;
        }
        if (relativistic_correction)
        {
            double delta_LT = light_travel_time(K1, f, w, ecc);
            double delta_TD = transverse_doppler(K1, f, ecc);
            double delta_GR = gravitational_redshift(K1, K2, f, ecc);
            v = v + delta_LT + delta_TD + delta_GR;
        }
        
    
        return v;
    }
}

// A solver for Kepler's equation based on:
//    Nijenhuis (1991)
//    http://adsabs.harvard.edu/abs/1991CeMDA..51..319N
// and
//    Markley (1995)
//    http://adsabs.harvard.edu/abs/1995CeMDA..63..101M
// Code from https://github.com/dfm/kepler.py
namespace nijenhuis
{
    // Implementation from numpy
    inline double npy_mod(double a, double b)
    {
        double mod = fmod(a, b);

        if (!b)
        {
            // If b == 0, return result of fmod. For IEEE is nan
            return mod;
        }

        // adjust fmod result to conform to Python convention of remainder
        if (mod)
        {
            if ((b < 0) != (mod < 0))
            {
                mod += b;
            }
        }
        else
        {
            // if mod is zero ensure correct sign
            mod = copysign(0, b);
        }

        return mod;
    }

    inline double get_markley_starter(double M, double ecc, double ome)
    {
        // M must be in the range [0, pi)
        const double FACTOR1 = 3 * M_PI / (M_PI - 6 / M_PI);
        const double FACTOR2 = 1.6 / (M_PI - 6 / M_PI);

        double M2 = M * M;
        double alpha = FACTOR1 + FACTOR2 * (M_PI - M) / (1 + ecc);
        double d = 3 * ome + alpha * ecc;
        double alphad = alpha * d;
        double r = (3 * alphad * (d - ome) + M2) * M;
        double q = 2 * alphad * ome - M2;
        double q2 = q * q;
        double w = pow(std::abs(r) + sqrt(q2 * q + r * r), 2.0 / 3);
        return (2 * r * w / (w * w + w * q + q2) + M) / d;
    }

    inline double refine_estimate(double M, double ecc, double ome, double E)
    {
        double sE = E - sin(E);
        double cE = 1 - cos(E);

        double f_0 = ecc * sE + E * ome - M;
        double f_1 = ecc * cE + ome;
        double f_2 = ecc * (E - sE);
        double f_3 = 1 - f_1;
        double d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1);
        double d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6);
        double d_42 = d_4 * d_4;
        double dE = -f_0 /
                    (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24);

        return E + dE;
    }

    /**
        Solve Kepler's equation for the eccentric anomaly

        @param M the mean anomaly
        @param ecc the orbital eccentricity
        @return E the eccentric anomaly
    */
    double solver(double M, double ecc)
    {
        // Wrap M into the range [0, 2*pi]
        M = npy_mod(M, TWO_PI);

        //
        bool high = M > M_PI;
        if (high)
            M = TWO_PI - M;

        // Get the starter
        double ome = 1.0 - ecc;
        double E = get_markley_starter(M, ecc, ome);

        // Refine this estimate using a high order Newton step
        E = refine_estimate(M, ecc, ome, E);

        if (high)
            E = TWO_PI - E;

        return E;
    }

    std::vector<double> solver(const std::vector<double> &M, double ecc)
    {
        std::vector<double> E(M.size());
        for (size_t i = 0; i < M.size(); i++)
            E[i] = solver(M[i], ecc);
        return E;
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
    double true_anomaly(double t, double period, double ecc, double t_peri)
    {
        double n = 2. * M_PI / period; // mean motion
        double M = n * (t - t_peri);   // mean anomaly

        // Solve Kepler's equation
        double E = solver(M, ecc);

        // Calculate true anomaly
        double cosE = cos(E);
        double f = acos((cosE - ecc) / (1 - ecc * cosE));
        // acos gives the principal values ie [0:PI]
        // when E goes above PI we need another condition
        if (E > M_PI)
            f = 2 * M_PI - f;

        return f;
    }

} // namespace nijenhuis


namespace brandt
{
    // Evaluate sine with a series expansion.  We can guarantee that the
    // argument will be <=pi/4, and this reaches double precision (within
    // a few machine epsilon) at a significantly lower cost than the
    // function call to sine that obeys the IEEE standard.
    double shortsin(const double &x)
    {
        const double if3 = 1. / 6;
        const double if5 = 1. / (6. * 20);
        const double if7 = 1. / (6. * 20 * 42);
        const double if9 = 1. / (6. * 20 * 42 * 72);
        const double if11 = 1. / (6. * 20 * 42 * 72 * 110);
        const double if13 = 1. / (6. * 20 * 42 * 72 * 110 * 156);
        const double if15 = 1. / (6. * 20 * 42 * 72 * 110 * 156 * 210);

        double x2 = x * x;
        return x *
               (1 - x2 * (if3 -
                          x2 * (if5 - x2 * (if7 - x2 * (if9 - x2 * (if11 - x2 * (if13 - x2 * if15)))))));
    }

    // Use the second-order series expanion in Raposo-Pulido & Pelaez (2017) in
    // the singular corner (eccentricity close to 1, mean anomaly close to zero)
    double EAstart(const double &M, const double &ecc)
    {
        const double ome = 1. - ecc;
        const double sqrt_ome = sqrt(ome);

        const double chi = M / (sqrt_ome * ome);
        const double Lam = sqrt(8 + 9 * chi * chi);
        const double S = cbrt(Lam + 3 * chi);
        const double sigma = 6 * chi / (2 + S * S + 4. / (S * S));
        const double s2 = sigma * sigma;
        const double s4 = s2 * s2;

        const double denom = 1.0 / (s2 + 2);
        const double E =
            sigma * (1 + s2 * ome * denom *
                             ((s2 + 20) / 60. +
                              s2 * ome * denom * denom * (s2 * s4 + 25 * s4 + 340 * s2 + 840) / 1400));

        return E * sqrt_ome;
    }


    double solver(const double &M, const double &ecc, double *sinE, double *cosE)
    {
        double bounds[13];
        double EA_tab[6*13];
        get_bounds(bounds, EA_tab, ecc);

        double E = solver_fixed_ecc(bounds, EA_tab, M, ecc, sinE, cosE);
        return E;
    }

    void get_bounds(double bounds[], double EA_tab[], double ecc)
    {
        const double pi = 3.14159265358979323846264338327950288;
        const double pi_d_12 = 3.14159265358979323846264338327950288 / 12;
        const double pi_d_6 = 3.14159265358979323846264338327950288 / 6;
        const double pi_d_4 = 3.14159265358979323846264338327950288 / 4;
        const double pi_d_3 = 3.14159265358979323846264338327950288 / 3;
        const double fivepi_d_12 = 3.14159265358979323846264338327950288 * 5. / 12;
        const double pi_d_2 = 3.14159265358979323846264338327950288 / 2;
        const double sevenpi_d_12 = 3.14159265358979323846264338327950288 * 7. / 12;
        const double twopi_d_3 = 3.14159265358979323846264338327950288 * 2. / 3;
        const double threepi_d_4 = 3.14159265358979323846264338327950288 * 3. / 4;
        const double fivepi_d_6 = 3.14159265358979323846264338327950288 * 5. / 6;
        const double elevenpi_d_12 = 3.14159265358979323846264338327950288 * 11. / 12;

        double g2s_e = 0.2588190451025207623489 * ecc;
        double g3s_e = 0.5 * ecc;
        double g4s_e = 0.7071067811865475244008 * ecc;
        double g5s_e = 0.8660254037844386467637 * ecc;
        double g6s_e = 0.9659258262890682867497 * ecc;
        double g2c_e = g6s_e;
        double g3c_e = g5s_e;
        double g4c_e = g4s_e;
        double g5c_e = g3s_e;
        double g6c_e = g2s_e;

        bounds[0] = 0;
        bounds[1] = pi_d_12 - g2s_e;
        bounds[2] = pi_d_6 - g3s_e;
        bounds[3] = pi_d_4 - g4s_e;
        bounds[4] = pi_d_3 - g5s_e;
        bounds[5] = fivepi_d_12 - g6s_e;
        bounds[6] = pi_d_2 - ecc;
        bounds[7] = sevenpi_d_12 - g6s_e;
        bounds[8] = twopi_d_3 - g5s_e;
        bounds[9] = threepi_d_4 - g4s_e;
        bounds[10] = fivepi_d_6 - g3s_e;
        bounds[11] = elevenpi_d_12 - g2s_e;
        bounds[12] = pi;

        double x;

        EA_tab[1] = 1 / (1. - ecc);
        EA_tab[2] = 0;

        x = 1. / (1 - g2c_e);
        EA_tab[7] = x;
        EA_tab[8] = -0.5 * g2s_e * x * x * x;
        x = 1. / (1 - g3c_e);
        EA_tab[13] = x;
        EA_tab[14] = -0.5 * g3s_e * x * x * x;
        x = 1. / (1 - g4c_e);
        EA_tab[19] = x;
        EA_tab[20] = -0.5 * g4s_e * x * x * x;
        x = 1. / (1 - g5c_e);
        EA_tab[25] = x;
        EA_tab[26] = -0.5 * g5s_e * x * x * x;
        x = 1. / (1 - g6c_e);
        EA_tab[31] = x;
        EA_tab[32] = -0.5 * g6s_e * x * x * x;

        EA_tab[37] = 1;
        EA_tab[38] = -0.5 * ecc;

        x = 1. / (1 + g6c_e);
        EA_tab[43] = x;
        EA_tab[44] = -0.5 * g6s_e * x * x * x;
        x = 1. / (1 + g5c_e);
        EA_tab[49] = x;
        EA_tab[50] = -0.5 * g5s_e * x * x * x;
        x = 1. / (1 + g4c_e);
        EA_tab[55] = x;
        EA_tab[56] = -0.5 * g4s_e * x * x * x;
        x = 1. / (1 + g3c_e);
        EA_tab[61] = x;
        EA_tab[62] = -0.5 * g3s_e * x * x * x;
        x = 1. / (1 + g2c_e);
        EA_tab[67] = x;
        EA_tab[68] = -0.5 * g2s_e * x * x * x;

        EA_tab[73] = 1. / (1 + ecc);
        EA_tab[74] = 0;

        double B0, B1, B2, idx;
        int i, k;
        for (i = 0; i < 12; i++)
        {
            idx = 1. / (bounds[i + 1] - bounds[i]);
            k = 6 * i;
            EA_tab[k] = i * pi_d_12;

            B0 = idx * (-EA_tab[k + 2] - idx * (EA_tab[k + 1] - idx * pi_d_12));
            B1 = idx * (-2 * EA_tab[k + 2] - idx * (EA_tab[k + 1] - EA_tab[k + 7]));
            B2 = idx * (EA_tab[k + 8] - EA_tab[k + 2]);

            EA_tab[k + 3] = B2 - 4 * B1 + 10 * B0;
            EA_tab[k + 4] = (-2 * B2 + 7 * B1 - 15 * B0) * idx;
            EA_tab[k + 5] = (B2 - 3 * B1 + 6 * B0) * idx * idx;
        }
    }

    // Calculate the eccentric anomaly, its sine and cosine, using a variant of
    // the algorithm suggested in Raposo-Pulido & Pelaez (2017) and used in
    // Brandt et al. (2020).  Use the series expansion above to generate an
    // initial guess in the singular corner and use a fifth-order polynomial to
    // get the initial guess otherwise.  Use series and square root calls to
    // evaluate sine and cosine, and update their values using series.  Accurate
    // to better than 1e-15 in E-ecc*sin(E)-M at all mean anomalies and at
    // eccentricies up to 0.999999.
    double solver_fixed_ecc(const double bounds[], const double EA_tab[],
                            const double &M, const double &ecc, double *sinE,
                            double *cosE)
    {
        const double one_sixth = 1. / 6;
        const double pi = 3.14159265358979323846264338327950288;
        const double pi_d_4 = 0.25 * pi;
        const double pi_d_2 = 0.5 * pi;
        const double threepi_d_4 = 0.75 * pi;
        const double twopi = 2 * pi;

        int j, k;
        double E = 0, MA, EA, sinEA, cosEA;
        double dx, num, denom, dEA, dEAsq_d6;
        double one_over_ecc = 1e17;
        if (ecc > 1e-17)
            one_over_ecc = 1. / ecc;

        int MAsign = 1;

        MA = mod2pi(M);
        if (MA > pi)
        {
            MAsign = -1;
            MA = twopi - MA;
        }

        if (ecc < 0.78)
        {
            // Use the lookup table for the initial guess.
            for (j = 11; j > 0; --j)
                if (MA > bounds[j])
                    break;

            k = 6 * j;
            dx = MA - bounds[j];
            EA =
                EA_tab[k] + dx * (EA_tab[k + 1] +
                                  dx * (EA_tab[k + 2] +
                                        dx * (EA_tab[k + 3] + 
                                            dx * (EA_tab[k + 4] + 
                                                dx * EA_tab[k + 5]))));

            // For sinEA, since _EA in [0,pi], sinEA should always be >=0 (no
            // sign ambiguity).  sqrt is much cheaper than sin.  If |cos|>|sin|,
            // compute them in reverse order, again using sqrt to avoid a trig
            // call.  Also, use trig identities, sin with a low argument and the
            // series expansion to minimize computational cost.

            if (EA <= pi_d_4)
            {
                sinEA = shortsin(EA);
                cosEA = sqrt(1 - sinEA * sinEA);
            }
            else if (EA < threepi_d_4)
            {
                cosEA = shortsin(pi_d_2 - EA);
                sinEA = sqrt(1 - cosEA * cosEA);
            }
            else
            {
                sinEA = shortsin(pi - EA);
                cosEA = -sqrt(1 - sinEA * sinEA);
            }

            num = (MA - EA) * one_over_ecc + sinEA;
            denom = one_over_ecc - cosEA;

            // Second order approximation.
            dEA = num * denom / (denom * denom + 0.5 * sinEA * num);

            // Apply our correction to EA, sinEA, and cosEA using
            // series.  Go to second order, since that was our level of
            // approximation above and will get us to basically machine
            // precision for eccentricities below 0.78.

            E = MAsign * (EA + dEA);
            *sinE = MAsign * (sinEA * (1 - 0.5 * dEA * dEA) + dEA * cosEA);
            *cosE = cosEA * (1 - 0.5 * dEA * dEA) - dEA * sinEA;
        }
        else
        {
            // Higher eccentricities will require a third-order correction to
            // achieve machine precision for all values of the eccentric
            // anomaly.  In the singular corner, they also use a series
            // expansion rather than the piecewise polynomial fit.

            if (2 * MA + (1 - ecc) > 0.2)
            {
                // Use the lookup table for the initial guess as long as we
                // are not in the singular corner.
                for (j = 11; j > 0; --j)
                    if (MA > bounds[j])
                        break;

                k = 6 * j;
                dx = MA - bounds[j];
                EA = EA_tab[k] +
                     dx * (EA_tab[k + 1] +
                           dx * (EA_tab[k + 2] +
                                 dx * (EA_tab[k + 3] + dx * (EA_tab[k + 4] + dx * EA_tab[k + 5]))));
            }
            else
            {
                // Use the series expansions in the singular corner.
                EA = EAstart(MA, ecc);
            }

            if (EA <= pi_d_4)
            {
                sinEA = shortsin(EA);
                cosEA = sqrt(1 - sinEA * sinEA);
            }
            else if (EA < threepi_d_4)
            {
                cosEA = shortsin(pi_d_2 - EA);
                sinEA = sqrt(1 - cosEA * cosEA);
            }
            else
            {
                sinEA = shortsin(pi - EA);
                cosEA = -sqrt(1 - sinEA * sinEA);
            }

            num = (MA - EA) * one_over_ecc + sinEA;
            denom = one_over_ecc - cosEA;

            if (MA > 0.4)
            {
                dEA = num * denom / (denom * denom + 0.5 * sinEA * num);
            }
            else
            {
                dEA = num * (denom * denom + 0.5 * num * sinEA);
                dEA /= denom * denom * denom + num * (denom * sinEA + one_sixth * num * cosEA);
            }

            dEAsq_d6 = dEA * dEA * one_sixth;

            // Apply our correction to EA, sinEA, and cosEA using series.  Go to
            // third order, since that was our level of approximation above and
            // will get us to basically machine precision at the higher
            // eccentricities.
            E = MAsign * (EA + dEA);
            *sinE = MAsign * (sinEA * (1 - 3 * dEAsq_d6) + dEA * (1 - dEAsq_d6) * cosEA);
            *cosE = cosEA * (1 - 3 * dEAsq_d6) - dEA * (1 - dEAsq_d6) * sinEA;
        }
        
        return mod2pi(E);
    }

    std::vector<double> solver(const std::vector<double> &M, double ecc)
    {
        double bounds[13];
        double EA_tab[6*13];
        get_bounds(bounds, EA_tab, ecc);

        std::vector<double> E(M.size());
        double sinE, cosE;
        for (size_t i = 0; i < M.size(); i++)
            E[i] = solver_fixed_ecc(bounds, EA_tab, M[i], ecc, &sinE, &cosE);
        return E;
    }

    void to_f(const double &ecc, const double &ome, double *sinf, double *cosf)
    {
        double denom = 1 + (*cosf);
        if (denom > 1.0e-10)
        {
            double tanf2 = sqrt((1 + ecc) / ome) * (*sinf) / denom; // tan(0.5*f)
            double tanf2_2 = tanf2 * tanf2;

            // Then we compute sin(f) and cos(f) using:
            // sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
            // cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
            denom = 1 / (1 + tanf2_2);
            *sinf = 2 * tanf2 * denom;
            *cosf = (1 - tanf2_2) * denom;
        }
        else
        {
            // If cos(E) = -1, E = pi and tan(0.5*E) -> inf and f = E = pi
            *sinf = 0;
            *cosf = -1;
        }
    }

    void solve_kepler(const double &M, const double &ecc, double *sinf,
                      double *cosf)
    {
        solver(M, ecc, sinf, cosf);
        to_f(ecc, 1 - ecc, sinf, cosf);
    }

    double true_anomaly(double t, double period, double ecc, double t_peri)
    {
        double n = 2. * M_PI / period; // mean motion
        double M = n * (t - t_peri);   // mean anomaly

        // Solve Kepler's equation
        double sinE, cosE;
        double E = solver(M, ecc, &sinE, &cosE);
        // std::cout << "brandt E:" << E << std::endl;

        // Calculate true anomaly
        double f = acos((cosE - ecc) / (1 - ecc * cosE));
        return f;
    }


    //
    std::vector<double> keplerian(const std::vector<double> &t, const double &P,
                                  const double &K, const double &ecc,
                                  const double &w, const double &M0,
                                  const double &M0_epoch)
    {
      // allocate RVs
      std::vector<double> rv(t.size());

      // mean motion, once per orbit
      double n = 2. * M_PI / P;
      // sin and cos of argument of periastron, once per orbit
      double sinw, cosw;
      __sincos(w, &sinw, &cosw);
      // ecentricity factor for g, once per orbit
      double g_e = sqrt((1 + ecc) / (1 - ecc));

      // brandt solver calculations, once per orbit
      double bounds[13];
      double EA_tab[6 * 13];
      get_bounds(bounds, EA_tab, ecc);

      for (size_t i = 0; i < t.size(); i++) {
        double E, sinE, cosE;
        double M = n * (t[i] - M0_epoch) - M0;
        E = solver_fixed_ecc(bounds, EA_tab, M, ecc, &sinE, &cosE);
        double g = g_e * ((1 - cosE) / sinE);
        double g2 = g * g;
        rv[i] = K * (cosw * ((1 - g2) / (1 + g2) + ecc) -
                     sinw * ((2 * g) / (1 + g2)));
      }

      return rv;
    }



}


// Solve Kepler's equation via the contour integration method described in
// Philcox et al. (2021). This uses techniques described in Ullisch (2020) to
// solve the `geometric goat problem'.
namespace contour
{
    // N_it specifies the number of grid-points.
    const int N_it = 10;

    void precompute_fft(const double &ecc, double exp2R[], double exp2I[],
                        double exp4R[], double exp4I[], double coshI[],
                        double sinhI[], double ecosR[], double esinR[],
                        double *esinRadius, double *ecosRadius) {
      double freq;
      // Define sampling points (actually use one more than this)
      int N_points = N_it - 2;
      int N_fft = (N_it - 1) * 2;

      // Define contour radius
      double radius = ecc / 2;

      // Generate e^{ikx} sampling points and precompute real and imaginary
      // parts
      for (int jj = 0; jj < N_points; jj++) {
        // NB: j = jj+1
        freq = 2.0 * M_PI * (jj + 1) / N_fft;
        exp2R[jj] = cos(freq);
        exp2I[jj] = sin(freq);
        exp4R[jj] = cos(2.0 * freq);
        exp4I[jj] = sin(2.0 * freq);
        coshI[jj] = cosh(radius * exp2I[jj]);
        sinhI[jj] = sinh(radius * exp2I[jj]);
        ecosR[jj] = ecc * cos(radius * exp2R[jj]);
        esinR[jj] = ecc * sin(radius * exp2R[jj]);
      }

      // Precompute e sin(e/2) and e cos(e/2)
      *esinRadius = ecc * sin(radius);
      *ecosRadius = ecc * cos(radius);
    }

    double solver_fixed_ecc(double exp2R[], double exp2I[], double exp4R[],
                            double exp4I[], double coshI[], double sinhI[],
                            double ecosR[], double esinR[],
                            const double &esinRadius, const double &ecosRadius,
                            const double &M, const double &ecc) {

        double E;
        double ft_gx2, ft_gx1, this_ell, freq, zR, zI, cosC, sinC, center;
        double fxR, fxI, ftmp, tmpcosh, tmpsinh, tmpcos, tmpsin;

        // Define sampling points (actually use one more than this)
        int N_points = N_it - 2;

        // Define contour radius
        double radius = ecc / 2;

        // Define contour center for each ell and precompute sin(center),
        // cos(center)
        if (M < M_PI)
          center = M + ecc / 2;
        else
          center = M - ecc / 2;
        sinC = sin(center);
        cosC = cos(center);
        E = center;

        // Accumulate Fourier coefficients
        // NB: we halve the range by symmetry, absorbing factor of 2 into ratio

        // Separate out j = 0 piece, which is simpler

        // Compute z in real and imaginary parts (zI = 0 here)
        zR = center + radius;

        // Compute e*sin(zR) from precomputed quantities
        tmpsin = sinC * ecosRadius + cosC * esinRadius; // sin(zR)

        // Compute f(z(x)) in real and imaginary parts (fxI = 0)
        fxR = zR - tmpsin - M;

        // Add to array, with factor of 1/2 since an edge
        ft_gx2 = 0.5 / fxR;
        ft_gx1 = 0.5 / fxR;

        ///////////////
        // Compute for j = 1 to N_points
        // NB: j = jj+1
        for (int jj = 0; jj < N_points; jj++) {

          // Compute z in real and imaginary parts
          zR = center + radius * exp2R[jj];
          zI = radius * exp2I[jj];

          // Compute f(z(x)) in real and imaginary parts
          // can use precomputed cosh / sinh / cos / sin for this!
          tmpcosh = coshI[jj];                          // cosh(zI)
          tmpsinh = sinhI[jj];                          // sinh(zI)
          tmpsin = sinC * ecosR[jj] + cosC * esinR[jj]; // e sin(zR)
          tmpcos = cosC * ecosR[jj] - sinC * esinR[jj]; // e cos(zR)

          fxR = zR - tmpsin * tmpcosh - M;
          fxI = zI - tmpcos * tmpsinh;

          // Compute 1/f(z) and append to array
          ftmp = fxR * fxR + fxI * fxI;
          fxR /= ftmp;
          fxI /= ftmp;

          ft_gx2 += (exp4R[jj] * fxR + exp4I[jj] * fxI);
          ft_gx1 += (exp2R[jj] * fxR + exp2I[jj] * fxI);
      }

      ///////////////
      // Separate out j = N_it piece, which is simpler

      // Compute z in real and imaginary parts (zI = 0 here)
      zR = center - radius;

      // Compute sin(zR) from precomputed quantities
      tmpsin = sinC * ecosRadius - cosC * esinRadius; // sin(zR)

      // Compute f(z(x)) in real and imaginary parts (fxI = 0 here)
      fxR = zR - tmpsin - M;

      // Add to sum, with 1/2 factor for edges
      ft_gx2 += 0.5 / fxR;
      ft_gx1 += -0.5 / fxR;

      // Compute E
      E += radius * ft_gx2 / ft_gx1;
      return E;
    }

    double solver(double M, double ecc)
    {
        double E;
        double ft_gx2, ft_gx1, this_ell, freq, zR, zI, cosC, sinC, esinRadius, ecosRadius, center;
        double fxR, fxI, ftmp, tmpcosh, tmpsinh, tmpcos, tmpsin;

        // Define sampling points (actually use one more than this)
        int N_points = N_it - 2;
        int N_fft = (N_it - 1) * 2;

        // Define contour radius
        double radius = ecc / 2;

        // Generate e^{ikx} sampling points and precompute real and imaginary parts
        double exp2R[N_points], exp2I[N_points], exp4R[N_points], exp4I[N_points], coshI[N_points], sinhI[N_points], ecosR[N_points], esinR[N_points];
        for (int jj = 0; jj < N_points; jj++)
        {
            // NB: j = jj+1
            freq = 2.0 * M_PI * (jj + 1) / N_fft;
            exp2R[jj] = cos(freq);
            exp2I[jj] = sin(freq);
            exp4R[jj] = cos(2.0 * freq);
            exp4I[jj] = sin(2.0 * freq);
            coshI[jj] = cosh(radius * exp2I[jj]);
            sinhI[jj] = sinh(radius * exp2I[jj]);
            ecosR[jj] = ecc * cos(radius * exp2R[jj]);
            esinR[jj] = ecc * sin(radius * exp2R[jj]);
        }

        // Precompute e sin(e/2) and e cos(e/2)
        esinRadius = ecc * sin(radius);
        ecosRadius = ecc * cos(radius);

        // Define contour center for each ell and precompute sin(center), cos(center)
        if (M < M_PI)
            center = M + ecc / 2;
        else
            center = M - ecc / 2;
        sinC = sin(center);
        cosC = cos(center);
        E = center;

        // Accumulate Fourier coefficients
        // NB: we halve the range by symmetry, absorbing factor of 2 into ratio

        ///////////////
        // Separate out j = 0 piece, which is simpler

        // Compute z in real and imaginary parts (zI = 0 here)
        zR = center + radius;

        // Compute e*sin(zR) from precomputed quantities
        tmpsin = sinC * ecosRadius + cosC * esinRadius; // sin(zR)

        // Compute f(z(x)) in real and imaginary parts (fxI = 0)
        fxR = zR - tmpsin - M;

        // Add to array, with factor of 1/2 since an edge
        ft_gx2 = 0.5 / fxR;
        ft_gx1 = 0.5 / fxR;

        ///////////////
        // Compute for j = 1 to N_points
        // NB: j = jj+1
        for (int jj = 0; jj < N_points; jj++)
        {

            // Compute z in real and imaginary parts
            zR = center + radius * exp2R[jj];
            zI = radius * exp2I[jj];

            // Compute f(z(x)) in real and imaginary parts
            // can use precomputed cosh / sinh / cos / sin for this!
            tmpcosh = coshI[jj];                          // cosh(zI)
            tmpsinh = sinhI[jj];                          // sinh(zI)
            tmpsin = sinC * ecosR[jj] + cosC * esinR[jj]; // e sin(zR)
            tmpcos = cosC * ecosR[jj] - sinC * esinR[jj]; // e cos(zR)

            fxR = zR - tmpsin * tmpcosh - M;
            fxI = zI - tmpcos * tmpsinh;

            // Compute 1/f(z) and append to array
            ftmp = fxR * fxR + fxI * fxI;
            fxR /= ftmp;
            fxI /= ftmp;

            ft_gx2 += (exp4R[jj] * fxR + exp4I[jj] * fxI);
            ft_gx1 += (exp2R[jj] * fxR + exp2I[jj] * fxI);
        }

        ///////////////
        // Separate out j = N_it piece, which is simpler

        // Compute z in real and imaginary parts (zI = 0 here)
        zR = center - radius;

        // Compute sin(zR) from precomputed quantities
        tmpsin = sinC * ecosRadius - cosC * esinRadius; // sin(zR)

        // Compute f(z(x)) in real and imaginary parts (fxI = 0 here)
        fxR = zR - tmpsin - M;

        // Add to sum, with 1/2 factor for edges
        ft_gx2 += 0.5 / fxR;
        ft_gx1 += -0.5 / fxR;

        // Compute E
        E += radius * ft_gx2 / ft_gx1;
        return E;
    }

    std::vector<double> solver(const std::vector<double> &M, double ecc)
    {
        double esinRadius, ecosRadius;
        // Define sampling points (actually use one more than this)
        int N_points = N_it - 2;
        double exp2R[N_points], exp2I[N_points], exp4R[N_points], exp4I[N_points], coshI[N_points], sinhI[N_points], ecosR[N_points], esinR[N_points];
        precompute_fft(ecc, exp2R, exp2I, exp4R, exp4I, coshI, sinhI, ecosR, esinR, &esinRadius, &ecosRadius);

        std::vector<double> E(M.size());
        for (size_t i = 0; i < M.size(); i++)
            E[i] = solver_fixed_ecc(exp2R, exp2I, exp4R, exp4I, coshI, sinhI, ecosR, esinR,
                                    esinRadius, ecosRadius, M[i], ecc);
        return E;
    }

}


