// (c) 2019 João Faria
// This file is part of kima, which is licensed under the MIT license (see LICENSE for details)

#include "Data.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <set>

using namespace std;

typedef vector <double> record_t;
typedef vector <record_t> data_t;


Data Data::instance;

Data::Data(){}

istream& operator >> ( istream& ins, record_t& record )
  // Let's overload the stream input operator to read a list of CSV fields (which a CSV record).
  //-----------------------------------------------------------------------------
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


istream& operator >> ( istream& ins, data_t& data )
// Overload the stream input operator to read a list of CSV records. We only 
// need to worry about reading records, and not fields.
{
  // make sure that the returned data only contains the CSV data we read here
  // data.clear();

  // for every record we can read from the file, append it to our resulting data
  // except if it's in the header
  record_t record;
  while (ins >> record)
    {
      data.push_back(record);
    }

  // return the argument stream
  return ins;  
}


/**
 * @brief Load RV data from a file.
 *
 * Read a tab/space separated file with columns  
 * ```
 *   time  vrad  error  
 *   ...   ...   ...
 * ```
 * 
 * @param filename   the name of the file
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file (default = 2)
 * @param indicators
*/
void Data::load(const std::string filename, const std::string units, 
                int skip, const vector<string>& indicators)
{

  data_t data;

  // Empty the vectors
  t.clear();
  y.clear();
  sig.clear();
  
  // check for indicator correlations and store stuff
  int nempty = count(indicators.begin(), indicators.end(), "");
  number_indicators = indicators.size() - nempty;
  indicator_correlations = number_indicators > 0;
  indicator_names = indicators;
  indicator_names.erase(
    std::remove( indicator_names.begin(), indicator_names.end(), "" ), 
    indicator_names.end() );


  // Empty the indicator vectors as well
  actind.clear();
  actind.resize(number_indicators);
  for (int n = 0; n < number_indicators; n++)
    actind[n].clear();

  // Read the file into the data container
  ifstream infile( filename );
  infile >> data;

  // Complain if something went wrong.
  if (!infile.eof())
  {
    printf("Could not read data file (%s)!\n", filename.c_str());
    exit(1);
  }

  infile.close();

  datafile = filename;
  dataunits = units;
  dataskip = skip;
  datamulti = false;
  number_instruments = 1;


  double factor = 1.;
  if(units == "kms") factor = 1E3;
  int j;

  for (size_t n = 0; n < data.size(); n++)
    {
      if (n < skip) continue;
      t.push_back(data[n][0]);
      y.push_back(data[n][1] * factor);
      sig.push_back(data[n][2] * factor);
  
      if (indicator_correlations)
      {
        j = 0;
        for (size_t i = 0; i < number_indicators + nempty; i++)
        {
          if (indicators[i] == "")
            continue; // skip column
          else
          {
            actind[j].push_back(data[n][3+i] * factor);
            j++;
          }
        }
      }
  
    }
  
  // epoch for the mean anomaly, by default the time of the first observation
  M0_epoch = t[0];


  // subtract means from activity indicators
  if (indicator_correlations)
  {
    double mean;
    for (auto& i: actind){ // use auto& instead of auto to modify i
      mean = std::accumulate(i.begin(), i.end(), 0.0) / y.size();
      std::for_each(i.begin(), i.end(), [mean](double& d) { d -= mean;});
    }
  }

  // How many points did we read?
  printf("# Loaded %zu data points from file %s\n", t.size(), filename.c_str());
  // Did we read activity indicators? how many?
  if(indicator_correlations){
    printf("# Loaded %zu observations of %zu activity indicators: ", 
           actind[0].size(), actind.size());
    for (const auto i: indicators){
      if (i != ""){
        printf("'%s'", i.c_str());
        (i != indicators.back()) ? cout << ", " : cout << " ";
      }
    }
    cout << endl;
  }
  // What are the units?
  if(units == "kms")
    printf("# Multiplied all RVs by 1000; units are now m/s.\n");

}


/**
 * @brief Load RV data from a multi-instrument file.
 * 
 * Read a tab/space separated file with columns  
 * ```
 *   time  vrad  error  obs
 *   ...   ...   ...    ...
 * ```
 * The `obs` column should be an integer identifying the instrument.
 * 
 * @param filename   the name of the file
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file (default = 2)
*/
void Data::load_multi(const std::string filename, const std::string units, int skip)
  {

  data_t data;

  // Empty the vectors
  t.clear();
  y.clear();
  sig.clear();
  obsi.clear();

  // Read the file into the data container
  ifstream infile( filename );
  infile >> data;

  // Complain if something went wrong.
  if (!infile.eof())
  {
    printf("Could not read data file (%s)!\n", filename.c_str());
    exit(1);
  }

  infile.close();

  datafile = filename;
  dataunits = units;
  dataskip = skip;
  datamulti = true;

  double factor = 1.;
  if(units == "kms") factor = 1E3;

  // the 4th column of the file identifies the instrument; can have "0"s
  // this is to make sure the obsi vector always starts at 1, to avoid
  // segmentation faults later
  vector<int> inst_id;
  int id_offset = 0;

  inst_id.push_back(data[skip][3]);

  for (size_t n = skip+1; n < data.size(); n++)
  {
    if (data[n][3] != inst_id.back())
      inst_id.push_back(data[n][3]);
  }

  id_offset = *min_element(inst_id.begin(), inst_id.end());

  for (unsigned n = skip; n < data.size(); n++)
  {
    // if (n < skip) continue;
    t.push_back(data[n][0]);
    y.push_back(data[n][1] * factor);
    sig.push_back(data[n][2] * factor);
    obsi.push_back(data[n][3] - id_offset + 1);
  }

  // How many points did we read?
  printf("# Loaded %zu data points from file %s\n", t.size(), filename.c_str());

  // Of how many instruments?
  std::set<int> s( obsi.begin(), obsi.end() );
  printf("# RVs come from %zu different instruments.\n", s.size());
  number_instruments = s.size();
  
  if(units == "kms") 
    cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

  // epoch for the mean anomaly, by default the time of the first observation
  M0_epoch = t[0];

  }



/**
 * @brief Load RV data from a multiple files.
 * 
 * Read a tab/space separated files, each with columns  
 * ```
 *   time  vrad  error
 *   ...   ...   ...
 * ```
 * All files should have the same structure and values in the same units.
 * 
 * @param filenames  the names of the files
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file (default = 2)
 * @param indicators
*/
void Data::load_multi(vector<std::string> filenames, const std::string units, 
                      int skip, const vector<std::string>& indicators)
{

  data_t data;

  // Empty the vectors
  t.clear();
  y.clear();
  sig.clear();
  obsi.clear();

  // check for indicator correlations and store stuff
  int nempty = count(indicators.begin(), indicators.end(), "");
  number_indicators = indicators.size() - nempty;
  indicator_correlations = number_indicators > 0;
  indicator_names = indicators;
  indicator_names.erase(
    std::remove( indicator_names.begin(), indicator_names.end(), "" ), 
    indicator_names.end() );

  // Empty the indicator vectors as well
  actind.clear();
  actind.resize(number_indicators);
  for (unsigned n = 0; n < number_indicators; n++)
    actind[n].clear();


  std::string dump; // to dump the first skip lines of each file
  int filecount = 1;
  int last_file_size = 0;

  // Read the files into the data container
  for (auto &filename : filenames) {
    ifstream infile( filename );
    for (int i=0; i<skip; i++)  // skip the first `skip` lines of each file
      getline(infile, dump);
    infile >> data;

    // Complain if something went wrong.
    if (!infile.eof())
    {
      printf("Could not read data file (%s)!\n", filename.c_str());
      exit(1);
    }

    infile.close();

    // Assign instrument int identifier to obsi
    for(unsigned i=last_file_size; i<data.size(); i++){
      obsi.push_back(filecount);
    }
    
    last_file_size = data.size();
    filecount++;
  }

  datafile = "";
  datafiles = filenames;
  dataunits = units;
  dataskip = skip;
  datamulti = true;


  double factor = 1.;
  if(units == "kms") factor = 1E3;

  for (unsigned n=0; n<data.size(); n++)
    {
      // if (n<skip) continue;
      t.push_back(data[n][0]);
      y.push_back(data[n][1] * factor);
      sig.push_back(data[n][2] * factor);

      if (indicator_correlations)
      {
        int j = 0;
        for (unsigned i = 0; i < number_indicators + nempty; i++)
        {
          if (indicators[i] == "")
            continue; // skip column
          else
          {
            actind[j].push_back(data[n][3+i] * factor);
            j++;
          }
        }
      }

    }

  // How many points did we read?
  printf("# Loaded %zu data points from files\n", t.size());
  cout << "#   ";
  for (auto f: filenames){
    cout << f.c_str();
    (f != filenames.back()) ? cout << " | " : cout << " ";
  }
  cout << endl;

  // Did we read activity indicators? how many?
  if(indicator_correlations){
    printf("# Loaded %d observations of %d activity indicators: ", actind[0].size(), actind.size());
    for (const auto i: indicators){
      if (i != ""){
        cout << "'" << i << "'";
        (i != indicators.back()) ? cout << ", " : cout << " ";
      }
    }
    cout << endl;
  }

  // Of how many instruments?
  std::set<int> s( obsi.begin(), obsi.end() );
  // set<int>::iterator iter;
  // for(iter=s.begin(); iter!=s.end();++iter) {  cout << (*iter) << endl;}
  printf("# RVs come from %zu different instruments.\n", s.size());
  number_instruments = s.size();

  if(units == "kms") 
    cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

  if(number_instruments > 1)
  {
    // We need to sort t because it comes from different instruments
    int N = t.size();
    std::vector<double> tt(N), yy(N);
    std::vector<double> sigsig(N), obsiobsi(N);
    std::vector<int> order(N);
    std::vector<std::vector<double>> actindactind(number_indicators, std::vector<double>(N));

    // order = argsort(t)
    int x=0;
    std::iota(order.begin(), order.end(), x++);
    sort( order.begin(),order.end(), [&](int i,int j){return t[i] < t[j];} );

    for(unsigned i=0; i<N; i++){
      tt[i] = t[order[i]];
      yy[i] = y[order[i]];
      sigsig[i] = sig[order[i]];
      obsiobsi[i] = obsi[order[i]];
      for(unsigned j=0; j<number_indicators; j++)
        actindactind[j][i] = actind[j][order[i]];
    }

    for(unsigned i=0; i<N; i++){
      t[i] = tt[i];
      y[i] = yy[i];
      sig[i] = sigsig[i];
      obsi[i] = obsiobsi[i];
      for(unsigned j=0; j<number_indicators; j++)
        actind[j][i] = actindactind[j][i];
    }

    // debug
    // for(std::vector<int>::size_type i = 0; i != t.size(); i++)
    //     cout << t[i] << "\t" << y[i] << "\t" << sig[i] << "\t" << obsi[i] <<  endl;
  }

  // epoch for the mean anomaly, by default the time of the first observation
  M0_epoch = t[0];

  }



double Data::get_RV_var() const
{
    double sum = std::accumulate(std::begin(y), std::end(y), 0.0);
    double mean =  sum / y.size();

    double accum = 0.0;
    std::for_each (std::begin(y), std::end(y), [&](const double d) {
        accum += (d - mean) * (d - mean);
    });
    return accum / (y.size()-1);
}



/**
 * @brief Calculate the maximum slope "allowed" by the data
 * 
 * This calculates peak-to-peak(RV) / peak-to-peak(time), which is a good upper
 * bound for the linear slope of a given dataset. When there are multiple 
 * instruments, the function returns the maximum of this peak-to-peak ratio of 
 * all individual instruments.
*/
double Data::topslope() const
{
  if (datamulti) {
    double slope = 0.0;
    for(size_t j=0; j<number_instruments; j++)
    {
      vector<double> obsy, obst;
      for (size_t i=0; i<y.size(); ++i) {
        if (obsi[i] == j + 1) {
          obsy.push_back(y[i]);
          obst.push_back(t[i]);
        }
      }
      const auto miny = std::min_element(obsy.begin(), obsy.end());
      const auto maxy = std::max_element(obsy.begin(), obsy.end());
      const auto mint = std::min_element(obst.begin(), obst.end());
      const auto maxt = std::max_element(obst.begin(), obst.end());
      double this_obs_topslope = (*maxy - *miny) / (*maxt - *mint);
      if (this_obs_topslope > slope)
        slope = this_obs_topslope;
    }
    return slope;
  }

  else {
    return get_RV_span() / get_timespan();
  }
}


/**
 * @brief Calculate the total span (peak to peak) of the radial velocities
*/
double Data::get_RV_span() const
{
  const auto min = std::min_element(y.begin(), y.end());
  const auto max = std::max_element(y.begin(), y.end());
  return *max - *min;
}

/**
 * @brief Calculate the maximum span (peak to peak) of the radial velocities
 * 
 * This is different from get_RV_span only in the case of multiple instruments: 
 * it returns the maximum of the spans of each instrument's RVs.
*/
double Data::get_max_RV_span() const
{
  // for multiple instruments, calculate individual RV spans and return largest
  if (datamulti) {
    double span = 0.0;
    for(size_t j=0; j<number_instruments; j++)
    {
      vector<double> obsy;
      for (size_t i=0; i<y.size(); ++i) 
      {
        if (obsi[i] == j + 1) 
        {
          obsy.push_back(y[i]);
        }
      }
      const auto min = std::min_element(obsy.begin(), obsy.end());
      const auto max = std::max_element(obsy.begin(), obsy.end());
      double this_obs_span = *max - *min;
      if (this_obs_span > span)
        span = this_obs_span;
    }
    return span;
  }

  // for one instrument only, this is easy
  else {
    return get_RV_span();
  }
}

double Data::get_adjusted_RV_var() const
{
    int ni;
    double sum, mean;
    std::vector<double> rva(t.size());

    for(size_t j=0; j<number_instruments; j++)
    {
      ni = 0;
      sum = 0.;
      for (size_t i=0; i<t.size(); i++)
        if(obsi[i] == j+1) {sum += y[i]; ni++;}
      mean = sum / ni;
      // cout << "sum: " << sum << endl;
      // cout << "mean: " << mean << endl;
      for (size_t i=0; i<t.size(); i++)
        if(obsi[i] == j+1) rva[i] = y[i] - mean;
    }

    mean = std::accumulate(rva.begin(), rva.end(), 0.0) / rva.size();
    double accum = 0.0;
    std::for_each (rva.begin(), rva.end(), [&](const double d) {
        accum += (d - mean) * (d - mean);
    });
    return accum / (y.size()-1);
}

/**
 * @brief Order of magnitude of trend coefficient (of degree) given the data
 * 
 * Returns the expected order of magnitude of the trend coefficient of degree 
 * `degree` supported by the data. It calculates the order of magnitude of 
 *    RVspan / timespan^degree
*/
int Data::get_trend_magnitude(int degree) const
{
  return (int)round(log10(get_RV_span() / pow(get_timespan(), degree)));
}

/**
 * @brief Find pathnames matching a pattern
 *
 * from https://stackoverflow.com/a/8615450
*/
std::vector<std::string> glob(const std::string& pattern) {
    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a vector<string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    return filenames;
}