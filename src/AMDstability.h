#pragma once

#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>

// gravitational constant G in AU**3 / (Msun * day**2), to the power of 1/3
#define G13 0.0666378476025686
// 1 Jupiter mass in Earth masses
#define mjup2mearth 317.8284065946748
// 1 Jupiter mass in solar masses
#define mjup2msun 0.0009545942339693249


#define STABLE 0
#define COLLISION 1
#define MMR_CIRCULAR 2
#define MMR_ECCENTRIC 3

using namespace std;

typedef vector<double> vd;

namespace AMD
{
    vector<size_t> argsort(const vd &array);
    int AMD_stable(const vector<vd>& components, double star_mass=1.0);
    double total_AMD_system(vd mu, vd a, vd e, vd i);
    vd am_circular(vd mu, vd a);
    int AMD_stability_pair(double mu1, double mu2, double a1, double a2, double Cx);
    double relative_AMD_collision(double mu1, double mu2, double a1, double a2);
    double critical_eccentricity(double y, double a);
    inline double F(double e, double y, double a);
    inline double dFde(double e, double y, double a);
    double relative_AMD_MMR_overlap(double mu1, double mu2, double a1, double a2);

}
