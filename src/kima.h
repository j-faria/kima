#include "DNest4.h"
#include "distributions/Fixed.h"

#include "Data.h"
#include "RVmodel.h"
#include "RVConditionalPrior.h"

const double PI = M_PI;

std::vector<char*> indicators;

using namespace DNest4;