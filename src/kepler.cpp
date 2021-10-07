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


namespace murison
{
    // A solver for Kepler's equation based on
    // "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006

    double kepler(double M, double ecc)
    {
        double tol;
        if (ecc < 0.8)
            tol = 1e-14;
        else
            tol = 1e-13;

        double Mnorm = fmod(M, 2. * M_PI);
        double E0 = keplerstart3(ecc, Mnorm);
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

    /**
        Calculates the eccentric anomaly at time t by solving Kepler's equation.
        See "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006

        @param t the time at which to calculate the eccentric anomaly.
        @param period the orbital period of the planet
        @param ecc the eccentricity of the orbit
        @param t_peri time of periastron passage
        @return eccentric anomaly.
    */
    double ecc_anomaly(double t, double period, double ecc, double time_peri)
    {
        double tol;
        if (ecc < 0.8)
            tol = 1e-14;
        else
            tol = 1e-13;

        double n = 2. * M_PI / period;  // mean motion
        double M = n * (t - time_peri); // mean anomaly
        double Mnorm = fmod(M, 2. * M_PI);
        double E0 = keplerstart3(ecc, Mnorm);
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

    /**
        Provides a starting value to solve Kepler's equation.
        See "A Practical Method for Solving the Kepler Equation", Marc A. Murison, 2006

        @param e the eccentricity of the orbit
        @param M mean anomaly (in radians)
        @return starting value for the eccentric anomaly.
    */
    double keplerstart3(double e, double M)
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
        // double E = solve_kepler(t, period, ecc, t_peri);
        double cosE = cos(E);
        double f = acos((cosE - ecc) / (1 - ecc * cosE));
        // acos gives the principal values ie [0:PI]
        // when E goes above PI we need another condition
        if (E > M_PI)
            f = 2 * M_PI - f;

        return f;
    }

} // namespace murison


// Code from https://github.com/dfm/kepler.py
namespace nijenhuis
{
    // A solver for Kepler's equation based on:
    //
    // Nijenhuis (1991)
    // http://adsabs.harvard.edu/abs/1991CeMDA..51..319N
    //
    // and
    //
    // Markley (1995)
    // http://adsabs.harvard.edu/abs/1995CeMDA..63..101M

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
    double kepler(double M, double ecc)
    {
        const double two_pi = 2 * M_PI;

        // Wrap M into the range [0, 2*pi]
        M = npy_mod(M, two_pi);

        //
        bool high = M > M_PI;
        if (high)
            M = two_pi - M;

        // Get the starter
        double ome = 1.0 - ecc;
        double E = get_markley_starter(M, ecc, ome);

        // Refine this estimate using a high order Newton step
        E = refine_estimate(M, ecc, ome, E);

        if (high)
            E = two_pi - E;

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
        double E = kepler(M, ecc);

        // Calculate true anomaly
        double cosE = cos(E);
        double f = acos((cosE - ecc) / (1 - ecc * cosE));
        // acos gives the principal values ie [0:PI]
        // when E goes above PI we need another condition
        if (E > M_PI)
            f = 2 * M_PI - f;

        return f;

    }

} // namespace kepler
