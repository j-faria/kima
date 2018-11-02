#include "Data.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>
#include <functional>

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
    //printf("--- t size = %i \n", t.size());
    for(unsigned i=0; i<data.size(); i++)
    {
        if (t[i] > 57170.)
        {
        index_fibers = i;
        break;
        }
    }
    create_y();
    create_fwhmerr();
    create_biserr();
    create_sig();
    create_tt();
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
void Data::create_fwhmerr()
{
    for(int n = 0; n < rverr.size(); n++)
    {
        fwhmerr.push_back(2.35 *rverr[n]);
    }
}


//BIS rms error
void Data::create_biserr() 
{
    for(int n = 0; n < rverr.size(); n++)
    {
        biserr.push_back(2.0 *rverr[n]);
    }

}


//to merge Rvs, fwhm, BIS and Rhk into a single vector
void Data::create_y()
{
    if((GP) && ((RN)))
    {
    y.reserve(rv.size() + fwhm.size() + bis.size() + rhk.size()); //preallocate memory
    y.insert(y.end(), rv.begin(), rv.end());
    y.insert(y.end(), fwhm.begin(), fwhm.end());
    y.insert(y.end(), bis.begin(), bis.end());
    y.insert(y.end(), rhk.begin(), rhk.end());
    }
    else
    {
    y = rv;
    }
}


//to merge all errors into a single vector
void Data::create_sig()
{
    if((GP) && ((RN)))
    {
    //merging RVs and the rest
    sig.reserve(rv.size() + fwhm.size() + bis.size() + rhk.size()); //preallocate memory
    sig.insert(sig.end(), rverr.begin(), rverr.end());
    sig.insert(sig.end(), fwhmerr.begin(), fwhmerr.end());
    sig.insert(sig.end(), biserr.begin(), biserr.end());
    sig.insert(sig.end(), rhkerr.begin(), rhkerr.end());
    }
    else
    {
    sig = rverr;

    }

}

//to merge all errors into a single vector
void Data::create_tt()
{
    if((GP) && ((RN)))
    {
    //merging RVs and the rest
    tt.reserve(4* t.size()); //preallocate memory
    tt.insert(tt.end(), t.begin(), t.end());
    tt.insert(tt.end(), t.begin(), t.end());
    tt.insert(tt.end(), t.begin(), t.end());
    tt.insert(tt.end(), t.begin(), t.end());
    }
    else
    {
    tt = t;

    }

}


