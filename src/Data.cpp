#include "Data.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>
#include <boost/lambda/lambda.hpp>

using namespace std;

typedef vector <double> record_t;
typedef vector <record_t> data_t;


Data Data::instance;

Data::Data(){}

//-----------------------------------------------------------------------------
// Let's overload the stream input operator to read a list of CSV fields (which a CSV record).
istream& operator >> ( istream& ins, record_t& record )
{
    // make sure that the returned record contains only the stuff we read now
    record.clear();

    // read the entire line into a string (a CSV record is terminated by a newline)
    string line;
    getline( ins, line );

    // now we'll use a stringstream to separate the fields out of the line
    stringstream ss( line );

    // convert each field to a double and
    // add the newly-converted field to the end of the record
    double f;
    while (ss >> f)
        record.push_back(f);

    // Now we have read a single line, converted into a list of fields, converted the fields
    // from strings to doubles, and stored the results in the argument record, so
    // we just return the argument stream as required for this kind of input overload function.
    return ins;
}

//-----------------------------------------------------------------------------
// Let's likewise overload the stream input operator to read a list of CSV records.
// This time it is a little easier, just because we only need to worry about reading
// records, and not fields.
istream& operator >> ( istream& ins, data_t& data )
{
    // used to hold the header
    data_t header;
    // make sure that the returned data only contains the CSV data we read here
    data.clear();

    // For every record we can read from the file, append it to our resulting data
    // except if it's in the header
    int i = 0;
    record_t record;
    while (ins >> record)
    {
        //if (i==0 || i==1) header.push_back(record);
        data.push_back(record);
        //  i++;
    }

    // Again, return the argument stream
    return ins;
}


void Data::load(const char* filename, const char* units, int skip)
    /*
    Read in tab/space separated file `filename` with columns
    jdb   vrad    svrad   fwhm    bis_span    rhk sig_rhk
    ---   ----    -----   ----    --------    --- -------
    where vrad and error are in `units` (either "kms" or "ms")
    */
{

    data_t data;

    // Empty the vectors
    t.clear();
    rv.clear();
    rverr.clear();
    fwhm.clear();
    bis.clear();
    rhk.clear();
    rhkerr.clear();

    // Read the file into the data container
    ifstream infile( filename );
    infile >> data;
    //operator>>(infile, data, skip);


    // Complain if something went wrong.
    if (!infile.eof())
    {
        printf("Could not read data file (%s)!\n", filename);
        exit(1);
    }

    infile.close();

    datafile = filename;
    dataunits = units;
    dataskip = skip;

    double factor = 1.;
    if(units == "kms") factor = 1E3;

    for (unsigned n = 0; n < data.size(); n++)
    {
        if (n<skip) continue;
        t.push_back(data[n][0]);
        rv.push_back(data[n][1] * factor);
        rverr.push_back(data[n][2] * factor);
        fwhm.push_back(data[n][3] * factor);
        bis.push_back(data[n][4] * factor);
        rhk.push_back(data[n][5]);
        rhkerr.push_back(data[n][6]);
    }

    // How many points did we read?
    printf("# Loaded %d data points from file %s\n", t.size(), filename);
    if(units == "kms") printf("# Multiplied all RVs and BIS by 1000; units are now m/s.\n");
    for(unsigned i=0; i<data.size(); i++)
    {
        if (t[i] > 57170.)
        {
        index_fibers = i;
        break;
        }
    }
    //create_y();
    //create_sig();
}



double Data::get_rv_var() const
{
    double sum = std::accumulate(std::begin(rv), std::end(rv), 0.0);
    double mean =  sum / rv.size();

    double accum = 0.0;
    std::for_each (std::begin(rv), std::end(rv), [&](const double d) {
        accum += (d - mean) * (d - mean);
    });
    return accum / (rv.size()-1);
}


////functor for getting sum of previous result and square of current element
//template<typename T>
//struct square
//{
//    T operator()(const T& Left, const T& Right) const
//    {
//        return (Left + Right*Right);
//    }
//};


//fwhm rms error
std::vector<double> Data::create_fwhmerr() const
{
//    double sum_of_elems;
//    double rms;
//    //sum of the squared elements
//    sum_of_elems = std::accumulate(fwhm.begin(), fwhm.end(), 0, square<int>() );
//    //sqrt(sum/n)
//    rms = std::sqrt(sum_of_elems / fwhm.size());
//    //now put values into vector
//    std::vector<double> fwhmerr(fwhm.size(), 0.1 * rms);
//    //std::cout << "myvector contains:";
//    //for (unsigned i=0; i<fwhmerr.size() ; i++)
//    //    std::cout << ' ' << fwhmerr[i];
//    //std::cout << '\n';
    std::vector<double> fwhmerr;
    fwhmerr.reserve(rverr.size()); //preallocate memory
    for(int n = 0; n < rverr.size(); n++)
    {
        fwhmerr.insert(fwhmerr.end(), 2.35 *rverr[n]);
    }
    return fwhmerr;
}


//BIS rms error
std::vector<double> Data::create_biserr() const
{
//    double sum_of_elems;
//    double rms;
//    //sum of the squared elements
//    sum_of_elems = std::accumulate(bis.begin(), bis.end(), 0, square<int>() );
//    //sqrt(sum/n)
//    rms = std::sqrt(sum_of_elems / bis.size());
//    //now put value into vector
//    std::vector<double> biserr(bis.size(), 0.2 * rms);
    std::vector<double> biserr;
    biserr.reserve(rverr.size()); //preallocate memory
    for(int n = 0; n < rverr.size(); n++)
    {
        biserr.insert(biserr.end(), 2.0 *rverr[n]);
    }
    return biserr;
}


//to merge Rvs, fwhm, BIS and Rhk into a single vector
std::vector<double> Data::create_y() const
{
    if((GP) && ((RN)))
    {
    printf("--- gprn \n");
    std::vector<double> y;
    y.reserve(rv.size() + fwhm.size() + bis.size() + rhk.size()); //preallocate memory
    y.insert(y.end(), rv.begin(), rv.end());
    y.insert(y.end(), fwhm.begin(), fwhm.end());
    y.insert(y.end(), bis.begin(), bis.end());
    y.insert(y.end(), rhk.begin(), rhk.end());
    }
    else
    {
    printf("--- not in gprn \n");
    std::vector<double> y = rv;
    printf("----- %i \n", y.size());
    }
    
    return y;
}


//to merge all errors into a single vector
std::vector<double> Data::create_sig() const
{
    if((GP) && ((RN)))
    {
    //merging RVs and the rest
    printf("--- we are here \n");
    std::vector<double> sig;
    sig.reserve(rv.size() + fwhm.size() + bis.size() + rhk.size()); //preallocate memory
    sig.insert(sig.end(), rverr.begin(), rverr.end());
    std::vector<double> fwhmerr = create_fwhmerr();
    sig.insert(sig.end(), fwhmerr.begin(), fwhmerr.end());
    std::vector<double> biserr = create_biserr();
    sig.insert(sig.end(), biserr.begin(), biserr.end());
    sig.insert(sig.end(), rhkerr.begin(), rhkerr.end());
    }
    else
    {
    printf("--- we are here!!!!! \n");
    std::vector<double> sig = rverr;
    printf("----- %i \n", sig.size());
    }
    return sig;
}


