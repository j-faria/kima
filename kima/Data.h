#pragma once

#include <glob.h>  // glob(), globfree()
#include <algorithm>
#include <cmath>
#include <cstring>  // memset()
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "loadtxt.hpp"

using namespace std;

// from https://stackoverflow.com/a/8615450
vector<string> glob(const string& pattern);


namespace kima {

class RVData {
   private:
    vector<double> t, y, sig, y2, sig2;
    vector<int> obsi;
    vector<vector<double>> actind;

    friend class RVmodel;

   public:
    RVData();
    friend ostream& operator<<(ostream& os, const RVData& d);

    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip = 2, 
              const string delimiter = " ",
              const vector<string> &indicators = vector<string>());

    // to read data from one file, more than one instrument
    void load_multi(const string filename, const string units, int skip = 2);

    // to read data from more than one file, more than one instrument
    void load_multi(vector<string> filenames, const string units, int skip = 2,
                    const vector<string>& indicators = vector<string>());

    bool indicator_correlations;
    int number_indicators;
    vector<string> indicator_names;

    string datafile;
    vector<string> datafiles;
    string dataunits;
    int dataskip;
    bool datamulti;  // multiple instruments? not sure if needed
    int number_instruments;

    /// docs for M0_epoch
    double M0_epoch;

    // to deprecate a function (C++14), put
    // [[deprecated("Replaced by bar, which has an improved interface")]]
    // before the definition

    /// Get the total number of RV points
    int N() const { return t.size(); }

    /// Get the array of times
    const vector<double>& get_t() const { return t; }

    /// Get the array of RVs
    const vector<double>& get_y() const { return y; }
    const vector<double>& get_y2() const { return y2; }
    /// Get the array of errors
    const vector<double>& get_sig() const { return sig; }
    const vector<double>& get_sig2() const { return sig2; }

    /// Get the mininum (starting) time
    double get_t_min() const { return *min_element(t.begin(), t.end()); }
    /// Get the maximum (ending) time
    double get_t_max() const { return *max_element(t.begin(), t.end()); }
    /// Get the timespan
    double get_timespan() const { return get_t_max() - get_t_min(); }
    double get_t_span() const { return get_t_max() - get_t_min(); }
    /// Get the middle time
    double get_t_middle() const { return get_t_min() + 0.5 * get_timespan(); }

    /// Get the mininum RV
    double get_RV_min() const { return *min_element(y.begin(), y.end()); }
    /// Get the maximum RV
    double get_RV_max() const { return *max_element(y.begin(), y.end()); }
    /// Get the RV span
    double get_RV_span() const;
    /// Get the maximum RV span
    double get_max_RV_span() const;
    /// Get the mean of the RVs
    double get_RV_mean() const;
    /// Get the variance of the RVs
    double get_RV_var() const;
    /// Get the standard deviation of the RVs
    double get_RV_std() const { return sqrt(get_RV_var()); }

    /// Get the mininum y2
    double get_y2_min() const { return *min_element(y2.begin(), y2.end()); }
    /// Get the maximum y2
    double get_y2_max() const { return *max_element(y2.begin(), y2.end()); }
    /// Get the y2 span
    double get_y2_span() const { return get_y2_max() - get_y2_min(); }

    /// Get the RV span, adjusted for multiple instruments
    double get_adjusted_RV_span() const;
    /// Get the RV variance, adjusted for multiple instruments
    double get_adjusted_RV_var() const;
    /// Get the RV standard deviation, adjusted for multiple instruments
    double get_adjusted_RV_std() const { return sqrt(get_adjusted_RV_var()); }

    /// Get the maximum slope allowed by the data
    double topslope() const;
    /// Order of magnitude of trend coefficient (of degree) given the data
    int get_trend_magnitude(int degree) const;

    /// Get the array of activity indictators
    const vector<vector<double>>& get_actind() const { return actind; }

    /// Get the array of instrument identifiers
    const vector<int>& get_obsi() const { return obsi; }

    /// Get the number of instruments
    int Ninstruments() const
    {
        set<int> s(obsi.begin(), obsi.end());
        return s.size();
    }

   private:
    // Singleton
    static RVData instance;

   public:
    static RVData& get_instance() { return instance; }
};


template <class... Args>
void load(Args&&... args)
{
    RVData::get_instance().load(args...);
}

template <class... Args>
void load_multi(Args&&... args)
{
    RVData::get_instance().load_multi(args...);
}

} // namespace kima