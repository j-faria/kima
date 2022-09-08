#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

double mod2pi(const double &angle);

namespace murison
{
    double solver(double M, double ecc);
    std::vector<double> solver(const std::vector<double> &M, double ecc);
    double ecc_anomaly(double t, double period, double ecc, double time_peri);
    double start3(double e, double M);
    double eps3(double e, double M, double x);
    double true_anomaly(double t, double period, double ecc, double t_peri);

    //
    std::vector<double> keplerian(const std::vector<double> &t, double P,
                                  double K, double ecc, double w, double M0,
                                  double M0_epoch);
}


namespace nijenhuis
{
    inline double npy_mod(double a, double b);
    inline double get_markley_starter(double M, double ecc, double ome);
    inline double refine_estimate(double M, double ecc, double ome, double E);
    double solver(double M, double ecc);
    std::vector<double> solver(const std::vector<double> &M, double ecc);
    double true_anomaly(double t, double period, double ecc, double t_peri);
}

namespace brandt
{
    double shortsin(const double &x);
    double EAstart(const double &M, const double &ecc);
    double solver(const double &M, const double &ecc, double *sinE, double *cosE);
    void get_bounds(double bounds[], double EA_tab[], double ecc);
    double solver_fixed_ecc(const double bounds[], const double EA_tab[],
                            const double &M, const double &ecc, double *sinE,
                            double *cosE);
    std::vector<double> solver(const std::vector<double> &M, double ecc);
    void to_f(const double &ecc, const double &ome, double *sinf, double *cosf);
    void solve_kepler(const double &M, const double &ecc, double *sinf,
                      double *cosf);
    double true_anomaly(double t, double period, double ecc, double t_peri);

    //
    std::vector<double> keplerian(const std::vector<double> &t, const double &P,
                                  const double &K, const double &ecc,
                                  const double &w, const double &M0,
                                  const double &M0_epoch);
}



namespace contour
{
    double solver(double M, double ecc);
    std::vector<double> solver(const std::vector<double> &M, double ecc);
    void precompute_fft(const double &ecc, double exp2R[], double exp2I[],
                        double exp4R[], double exp4I[], double coshI[],
                        double sinhI[], double ecosR[], double esinR[],
                        double *esinRadius, double *ecosRadius);
    double solver_fixed_ecc(double exp2R[], double exp2I[], double exp4R[],
                            double exp4I[], double coshI[], double sinhI[],
                            double ecosR[], double esinR[],
                            const double &esinRadius, const double &ecosRadius,
                            const double &M, const double &ecc);
}

