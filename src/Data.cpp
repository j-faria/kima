#include "Data.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>

using namespace std;

typedef vector <double> record_t;
typedef vector <record_t> data_t;


Data Data::instance;

Data::Data()
{

}

void Data::load(const char* filename)
{
  fstream fin(filename, ios::in);
  if(!fin)
  {
    cerr<<"# Error. Couldn't open file "<<filename<<endl;
    return;
  }

  // Empty the vectors
  t.clear();
  y.clear();
  sig.clear();

  int it = 0;
  double temp1, temp2, temp3;
  while(fin>>temp1 && fin>>temp2 && fin>>temp3)
  {
    /*if (it==0 || it==1) {
      it++;
      continue;
    }*/
    t.push_back(temp1);
    y.push_back(temp2);
    sig.push_back(temp3);
    it++;
  }
  cout<<"# Loaded "<<t.size()<<" data points from file "
      <<filename<<endl;
  fin.close();
  //cout<<it<<endl;

  double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
  //cout<<mean<<endl;

  // this is probably a stupid way to substract the mean and convert to m/s
  std::transform( y.begin(), y.end(), y.begin(), std::bind2nd( minus<double>(), mean ) );
  std::transform( y.begin(), y.end(), y.begin(), std::bind2nd( multiplies<double>(), 1000. ) );
  std::transform( y.begin(), y.end(), y.begin(), std::bind2nd( plus<double>(), mean ) );

  // the errorbars just need to be converted to m/s
  std::transform( sig.begin(), sig.end(), sig.begin(), std::bind2nd( multiplies<double>(), 1000. ) );
  
  //for (std::vector<double>::const_iterator i = sig.begin(); i != sig.end(); ++i)
    //std::cout << *i << '\n';
  //std::cout << '\n';
}



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
      if (i==0 || i==1) header.push_back(record);
      else data.push_back(record);
        i++;
    }

  // Again, return the argument stream
  return ins;  
  }


void Data::loadnew(const char* filename, const char* fileunits)
  /* 
  Read in tab/space separated file `filename` with columns
  time  vrad  error
  where vrad and error are in units `fileunits` (either "kms" or "ms")
  */
  {
  // Here is the data we want.
  data_t data;

  // Empty the vectors
  t.clear();
  y.clear();
  sig.clear();

  // Here is the file containing the data. Read it into data.
  ifstream infile( filename );
  infile >> data;

  // Complain if something went wrong.
  if (!infile.eof())
    {
    cout << "Fooey!\n";
    }

  infile.close();

  // Otherwise, list some basic information about the file.
  cout << "# Loaded " << data.size() << " data points from file "
                 <<filename<<endl;

  double factor = 1.;
  if(fileunits == "kms") factor = 1E3;
  

  for (unsigned n = 0; n < data.size(); n++)
    {
      t.push_back(data[n][0]);
      y.push_back(data[n][1] * factor);
      sig.push_back(data[n][2] * factor);
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