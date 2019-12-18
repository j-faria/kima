#include "DNest4.h"
#include "distributions/Fixed.h"
#include "distributions/Empirical.h"

#include "Data.h"
#include "RVmodel.h"
#include "RVConditionalPrior.h"

const double PI = M_PI;

std::string datafile;
std::vector<std::string> datafiles;


using ind = std::vector<char*>;
std::vector<std::string> indicators;


using namespace DNest4;