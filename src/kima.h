#pragma once

#include "DNest4.h"
#include "distributions/Fixed.h"
#include "distributions/Empirical.h"
#include "distributions/mixGaussianLogUniform.h"
#include "distributions/Gaussian_from_Tc.h"
#include "distributions/InvGamma.h"

#include "Data.h"
#include "ConditionalPrior.h"
#include "RVmodel.h"
#include "RVFWHMmodel.h"

const double PI = M_PI;

std::string datafile;
std::vector<std::string> datafiles;


using ind = std::vector<char*>;
std::vector<std::string> indicators;


using namespace DNest4;