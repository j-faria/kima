#pragma once

#include "DNest4.h"

#include "distributions/distributions.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "GP.h"
#include "RVmodel.h"
#include "GPmodel.h"
#include "GPmodel2.h"
#include "RVFWHMmodel.h"
#include "BDmodel.h"
#include "BINARIESmodel.h"


using namespace DNest4;
using namespace kima;

const double PI = M_PI;

std::string datafile;
std::vector<std::string> datafiles;

using ind = std::vector<char*>;
std::vector<std::string> indicators;

