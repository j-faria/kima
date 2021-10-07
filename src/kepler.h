#pragma once

#include <cmath>
#include <vector>
#include <execution>
#include <algorithm>
#include <iostream>

double mod2pi(const double &angle);

namespace murison
{
    double solver(double M, double ecc);
    std::vector<double> solver(std::vector<double> M, double ecc);
    double ecc_anomaly(double t, double period, double ecc, double time_peri);
    double start3(double e, double M);
    double eps3(double e, double M, double x);
    double true_anomaly(double t, double period, double ecc, double t_peri);

    //
    std::vector<double> keplerian(std::vector<double> t, const double &P,
                                const double &K, const double &ecc,
                                const double &w, const double &M0,
                                const double &M0_epoch);

}


namespace nijenhuis
{
    inline double npy_mod(double a, double b);
    inline double get_markley_starter(double M, double ecc, double ome);
    inline double refine_estimate(double M, double ecc, double ome, double E);
    double solver(double M, double ecc);
    std::vector<double> solver(std::vector<double> M, double ecc);
    double true_anomaly(double t, double period, double ecc, double t_peri);
}


namespace murison
{
    double kepler(double M, double ecc);
    double ecc_anomaly(double t, double period, double ecc, double time_peri);
    double keplerstart3(double e, double M);
    double eps3(double e, double M, double x);
    double true_anomaly(double t, double period, double ecc, double t_peri);

namespace contour
{
    double solver(double M, double ecc);
    std::vector<double> solver(std::vector<double> M, double ecc);
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

