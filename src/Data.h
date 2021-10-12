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

using namespace std;

// from https://stackoverflow.com/a/8615450
vector<string> glob(const string& pattern);

typedef vector<double> vd;
typedef vector<double> record_t;
typedef vector<record_t> data_t;

class RVData {
   private:
    vector<double> t, y, sig, y2, sig2;
    vector<int> obsi;
    vector<vector<double>> actind;

   public:
    RVData();

    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip = 2,
              const vector<string>& indicators = vector<string>());

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

    /// @brief Get the array of times @return const vector<double>&
    const vector<double>& get_t() const { return t; }

    /// @brief Get the array of RVs @return const vector<double>&
    const vector<double>& get_y() const { return y; }
    const vector<double>& get_y2() const { return y2; }
    /// Get the array of errors @return const vector<double>&
    const vector<double>& get_sig() const { return sig; }
    const vector<double>& get_sig2() const { return sig2; }

    /// @brief Get the mininum (starting) time @return double
    double get_t_min() const { return *min_element(t.begin(), t.end()); }
    /// @brief Get the maximum (ending) time @return double
    double get_t_max() const { return *max_element(t.begin(), t.end()); }
    /// @brief Get the timespan @return double
    double get_timespan() const { return get_t_max() - get_t_min(); }
    double get_t_span() const { return get_t_max() - get_t_min(); }
    /// @brief Get the middle time @return double
    double get_t_middle() const { return get_t_min() + 0.5 * get_timespan(); }

    /// @brief Get the mininum RV @return double
    double get_RV_min() const { return *min_element(y.begin(), y.end()); }
    /// @brief Get the maximum RV @return double
    double get_RV_max() const { return *max_element(y.begin(), y.end()); }
    /// @brief Get the RV span @return double
    double get_RV_span() const;
    /// @brief Get the maximum RV span @return double
    double get_max_RV_span() const;
    /// @brief Get the variance of the RVs @return double
    double get_RV_var() const;
    /// @brief Get the standard deviation of the RVs @return double
    double get_RV_std() const { return sqrt(get_RV_var()); }

    /// @brief Get the mininum y2 @return double
    double get_y2_min() const { return *min_element(y2.begin(), y2.end()); }
    /// @brief Get the maximum y2 @return double
    double get_y2_max() const { return *max_element(y2.begin(), y2.end()); }
    /// @brief Get the y2 span @return double
    double get_y2_span() const { return get_y2_max() - get_y2_min(); }

    /// @brief Get the RV span, adjusted for multiple instruments @return double
    double get_adjusted_RV_span() const;
    /// @brief Get the RV variance, adjusted for multiple instruments @return
    /// double
    double get_adjusted_RV_var() const;
    /// @brief Get the RV standard deviation, adjusted for multiple instruments
    /// @return double
    double get_adjusted_RV_std() const { return sqrt(get_adjusted_RV_var()); }

    /// @brief Get the maximum slope allowed by the data. @return double
    double topslope() const;
    /// @brief Order of magnitude of trend coefficient (of degree) given the
    /// data
    int get_trend_magnitude(int degree) const;

    /// @brief Get the array of activity indictators @return
    /// vector<vector<double>>&
    const vector<vector<double>>& get_actind() const { return actind; }

    /// @brief Get the array of instrument identifiers @return vector<int>&
    const vector<int>& get_obsi() const { return obsi; }

    /// @brief Get the number of instruments. @return int
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



class LCData {
   private:
    vector<double> t, flux, sig;
    // vector<int> obsi;

    friend class TRANSITmodel;

   public:
    LCData();

    // to read data from one file, one instrument
    void load(const string filename, int skip = 0);

    string datafile;
    vector<string> datafiles;
    string dataunits = "";
    int dataskip;
    bool datamulti;  // multiple instruments? not sure if needed
    int number_instruments = 1;

    /// docs for M0_epoch
    double M0_epoch;

    // to deprecate a function (C++14), put
    // [[deprecated("Replaced by bar, which has an improved interface")]]
    // before the definition

    /// Get the total number of RV points
    int N() const { return t.size(); }

    /// @brief Get the array of times @return const vector<double>&
    const vector<double>& get_t() const { return t; }

    /// @brief Get the array of flux @return const vector<double>&
    const vector<double>& get_flux() const { return flux; }
    /// Get the array of errors @return const vector<double>&
    const vector<double>& get_sig() const { return sig; }

    /// @brief Get the mininum (starting) time @return double
    double get_t_min() const { return *min_element(t.begin(), t.end()); }
    /// @brief Get the maximum (ending) time @return double
    double get_t_max() const { return *max_element(t.begin(), t.end()); }
    /// @brief Get the timespan @return double
    double get_timespan() const { return get_t_max() - get_t_min(); }
    double get_t_span() const { return get_t_max() - get_t_min(); }
    /// @brief Get the middle time @return double
    double get_t_middle() const { return get_t_min() + 0.5 * get_timespan(); }

    /// @brief Get the mininum flux @return double
    double get_flux_min() const { return *min_element(flux.begin(), flux.end()); }
    /// @brief Get the maximum flux @return double
    double get_flux_max() const { return *max_element(flux.begin(), flux.end()); }
    /// @brief Get the flux span @return double
    double get_flux_span() const;
    /// @brief Get the variance of the flux @return double
    double get_flux_var() const;
    /// @brief Get the standard deviation of the flux @return double
    double get_flux_std() const { return sqrt(get_flux_var()); }


    /// @brief Get the maximum slope allowed by the data. @return double
    double topslope() const;
    /// @brief Order of magnitude of trend coefficient (of degree) given the
    /// data
    int get_trend_magnitude(int degree) const;

   private:
    // Singleton
    static LCData LCinstance;

   public:
    static LCData& get_instance() { return LCinstance; }
};


template <class... Args>
void load_lc(Args&&... args)
{
    LCData::get_instance().load(args...);
}
