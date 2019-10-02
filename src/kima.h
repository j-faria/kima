#include "DNest4.h"
#include "distributions/Fixed.h"
#include "distributions/Empirical.h"

#include "Data.h"
#include "RVmodel.h"
#include "RVConditionalPrior.h"

const double PI = M_PI;

char* datafile;
std::vector<char*> datafiles;


using ind = std::vector<char*>;
std::vector<char*> indicators;


using namespace DNest4;