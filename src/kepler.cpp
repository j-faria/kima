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

    std::vector<double> solver(std::vector<double> M, double ecc)
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
    std::vector<double> keplerian(std::vector<double> t, const double &P,
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
        sincos(w, &sinw, &cosw);
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

    std::vector<double> solver(std::vector<double> M, double ecc)
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

    std::vector<double> solver(std::vector<double> M, double ecc)
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

