#include "Empirical.h"

namespace DNest4
{

/**
 * Construct an Empirical distribution from samples stored in a file.
 * 
 * `filename` should contain *one sample per line*
*/
Empirical::Empirical(const char* filename)
{
    string value;
	ifstream file(filename);
    while(file >> value) {
        data.push_back(stod(value));
    }
	file.close();
    // std::cout << "done reading data from " << filename << std::endl;
    std::sort(data.begin(), data.end());
}

/* Construct an Empirical distribution from samples stored in a vector. */
Empirical::Empirical(const vector<double> datain)
{
    data = datain;
    std::sort(data.begin(), data.end());
}

/* Empirical distribution function */
double Empirical::cdf(double x) const
{
    auto upper = std::upper_bound(data.begin(), data.end(), x);
    return (upper - data.begin()) / double(data.size());
}

/* Quantile estimate, midpoint interpolation */
double Empirical::cdf_inverse(double p) const
{
    if(p < 0.0 || p > 1.0)
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
    
    double poi = -0.5*(1 - p) + p*(data.size()-0.5);

    size_t left = max(int64_t(floor(poi)), int64_t(0));
    size_t right = min(int64_t(ceil(poi)), int64_t(data.size() - 1));

    double datLeft = data.at(left);
    double datRight = data.at(right);

    double t = poi - left;
    double quantile = (1 - t)*datLeft + t*datRight;

    return quantile;
}

double Empirical::log_pdf(double x) const
{
    // not implemented
}



} // namespace DNest4

