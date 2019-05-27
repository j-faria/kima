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
  // Let's likewise overload the stream input operator to read a list of CSV records.
  // This time it is a little easier, just because we only need to worry about reading
  // records, and not fields.
  //-----------------------------------------------------------------------------
  {
  // make sure that the returned data only contains the CSV data we read here
  // data.clear();

  // For every record we can read from the file, append it to our resulting data
  // except if it's in the header
  int i = 0;
  record_t record;
  while (ins >> record)
    {
      data.push_back(record);
    }

  // Again, return the argument stream
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
void Data::load(const char* filename, const char* units, 
                int skip, const vector<char*>& indicators)
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
  for (unsigned n = 0; n < number_indicators; n++)
    actind[n].clear();

  // Read the file into the data container
  ifstream infile( filename );
  infile >> data;

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
  datamulti = false;
  number_instruments = 1;


  double factor = 1.;
  if(units == "kms") factor = 1E3;
  int j;

  for (unsigned n = 0; n < data.size(); n++)
    {
      if (n<skip) continue;
      t.push_back(data[n][0]);
      y.push_back(data[n][1] * factor);
      sig.push_back(data[n][2] * factor);
  
      if (indicator_correlations)
      {
        j = 0;
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
  printf("# Loaded %d data points from file %s\n", t.size(), filename);
  // Did we read activity indicators? how many?
  if(indicator_correlations){
    printf("# Loaded %d observations of %d activity indicators: ", actind[0].size(), actind.size());
    for (const auto i: indicators){
      if (i != ""){
        printf("'%s'", i);
        (i != indicators.back()) ? cout << ", " : cout << " ";
      }
    }
    cout << endl;
  }
  // What are the units?
  if(units == "kms")
    printf("# Multiplied all RVs by 1000; units are now m/s.\n");

  for(unsigned i=0; i<data.size(); i++)
  {
      if (t[i] > 57170.)
      {
          index_fibers = i;
          break;
      }
  }

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
void Data::load_multi(const char* filename, const char* units, int skip)
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
    printf("Could not read data file (%s)!\n", filename);
    exit(1);
  }

  infile.close();

  datafile = filename;
  dataunits = units;
  dataskip = skip;
  datamulti = true;

  double factor = 1.;
  if(units == "kms") factor = 1E3;

  for (unsigned n = 0; n < data.size(); n++)
    {
      if (n<skip) continue;
      t.push_back(data[n][0]);
      y.push_back(data[n][1] * factor);
      sig.push_back(data[n][2] * factor);
      obsi.push_back(data[n][3]);
    }

  // How many points did we read?
  printf("# Loaded %d data points from file %s\n", t.size(), filename);

  // Of how many instruments?
  std::set<int> s( obsi.begin(), obsi.end() );
  printf("# RVs come from %d different instruments.\n", s.size());
  number_instruments = s.size();
  
  if(units == "kms") 
    cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

  for(unsigned i=0; i<data.size(); i++)
  {
      if (t[i] > 57170.)
      {
          index_fibers = i;
          break;
      }
  }

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
void Data::load_multi(vector<char*> filenames, const char* units, int skip,
                      const vector<char*>& indicators)
{

  data_t data;

  // Empty the vectors
  t.clear();
  y.clear();
  sig.clear();
  obsi.clear();

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
      printf("Could not read data file (%s)!\n", filename);
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
    }

  // How many points did we read?
  printf("# Loaded %d data points from files\n", t.size());
  cout << "# ";
  for (auto f: filenames)
    cout << f << " ; ";
  cout << endl;

  // Of how many instruments?
  std::set<int> s( obsi.begin(), obsi.end() );
  // set<int>::iterator iter;
  // for(iter=s.begin(); iter!=s.end();++iter) {  cout << (*iter) << endl;}
  printf("# RVs come from %d different instruments.\n", s.size());
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

    // order = argsort(t)
    int x=0;
    std::iota(order.begin(), order.end(), x++);
    sort( order.begin(),order.end(), [&](int i,int j){return t[i] < t[j];} );

    for(unsigned i=0; i<N; i++){
      tt[i] = t[order[i]];
      yy[i] = y[order[i]];
      sigsig[i] = sig[order[i]];
      obsiobsi[i] = obsi[order[i]];
    }

    for(unsigned i=0; i<N; i++){
      t[i] = tt[i];
      y[i] = yy[i];
      sig[i] = sigsig[i];
      obsi[i] = obsiobsi[i];
    }

    // debug
    // for(std::vector<int>::size_type i = 0; i != t.size(); i++)
    //     cout << t[i] << "\t" << y[i] << "\t" << sig[i] << "\t" << obsi[i] <<  endl;
  }

  for(unsigned i=0; i<data.size(); i++)
  {
      if (t[i] > 57170.)
      {
          index_fibers = i;
          break;
      }
  }

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


double Data::get_adjusted_RV_span() const
/* Return the span of the RVs, after subtracting the mean for each instrument */
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

    double min = *std::min_element(rva.begin(), rva.end());
    double max = *std::max_element(rva.begin(), rva.end());
    return max - min;
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
