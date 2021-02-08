#pragma once

#include <math.h>

namespace kepler
{
    inline double npy_mod(double a, double b);
    inline double get_markley_starter(double M, double ecc, double ome);
    inline double refine_estimate(double M, double ecc, double ome, double E);
    double kepler(double M, double ecc);
    double true_anomaly(double t, double period, double ecc, double t_peri);
}


namespace murison
{
    double kepler(double M, double ecc);
    double ecc_anomaly(double t, double period, double ecc, double time_peri);
    double keplerstart3(double e, double M);
    double eps3(double e, double M, double x);
    double true_anomaly(double t, double period, double ecc, double t_peri);
}

