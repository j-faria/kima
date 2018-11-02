#ifndef DNest4_Data
#define DNest4_Data

#include <vector>
#include <algorithm>
#include <cmath>

// whether the model includes a GP component
extern const bool GP;

// whether the model is to be a GPRN
extern const bool RN;

class Data
{
    private:
        std::vector<double> t, rv, rverr, fwhm, fwhmerr, bis, biserr, 
                                                        rhk, rhkerr, tt, y, sig;

    public:
        Data();
        //void load(const char* filename);
        void load(const char* filename, const char* units, int skip=2);
        int index_fibers;

        const char* datafile;
        const char* dataunits;
        int dataskip;

        // Getters
        int N() const {return t.size();}

        //time
        const std::vector<double>& get_t() const { return t; }
        double get_t_min() const { return *std::min_element(t.begin(), t.end()); }
        double get_t_max() const { return *std::max_element(t.begin(), t.end()); }
        double get_t_middle() const { return get_t_min() + 0.5*(get_t_max() - get_t_min()); }
        double get_timespan() const { return get_t_max() - get_t_min(); }

        //RVs
        const std::vector<double>& get_rv() const { return rv; }
        double get_rv_min() const { return *std::min_element(rv.begin(), rv.end()); }
        double get_rv_max() const { return *std::max_element(rv.begin(), rv.end()); }
        double get_rv_span() const { return get_rv_max() - get_rv_min(); }
        double get_rv_var() const;
        double get_vr_std() const { return std::sqrt(get_rv_var()); }

        //RVs error
        const std::vector<double>& get_rverr() const { return rverr; }

        //slope
        double topslope() const {return std::abs(get_rv_max() - get_rv_min()) / (t.back() - t.front());}

        //The fwhm, BIS, Rhk and respective errors
        const std::vector<double>& get_fwhm() const { return fwhm; }
        void create_fwhmerr();
        const std::vector<double>& get_fwhmerr() const { return fwhmerr; }

        const std::vector<double>& get_bis() const { return bis; }
        void create_biserr();
        const std::vector<double>& get_biserr() const {return biserr; }
        
        const std::vector<double>& get_rhk() const { return rhk; }
        const std::vector<double>& get_rhkerr() const { return rhkerr; }

        //single vector of tt, y and sig
        void create_tt();
        const std::vector<double>& get_tt() const {return tt; }
        void create_y();
        const std::vector<double>& get_y() const {return y; }
        void create_sig();
        const std::vector<double>& get_sig() const {return sig; }
        
    // Singleton
    private:
        static Data instance;

    public:
        static Data& get_instance() { return instance; }
};

#endif

