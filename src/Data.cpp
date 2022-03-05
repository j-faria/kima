#include "Data.h"

#ifndef VERBOSE
#define VERBOSE true
#endif

RVData RVData::instance;
RVData::RVData() {}

// overload the stream input operator to read a list of CSV fields
istream& operator>>(istream& ins, record_t& record)
{
    // make sure that the returned record contains only the stuff we read now
    record.clear();

    // read the entire line into a string (a CSV record is terminated by a
    // newline)
    string line;
    getline(ins, line);

    // now we'll use a stringstream to separate the fields out of the line
    stringstream ss(line);

    // convert each field to a double and
    // add the newly-converted field to the end of the record
    double f;
    while (ss >> f) record.push_back(f);

    // Now we have read a single line, converted into a list of fields,
    // converted the fields from strings to doubles, and stored the results in
    // the argument record, so we just return the argument stream as required
    // for this kind of input overload function.
    return ins;
}

// Overload the stream input operator to read a list of CSV records. We only
// need to worry about reading records, and not fields.
istream& operator>>(istream& ins, data_t& data)
{
    // make sure that the returned data only contains the CSV data we read here
    // data.clear();

    // for every record we can read from the file, append it to our resulting
    // data except if it's in the header
    record_t record;
    while (ins >> record) {
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
 *   time  vrad  error  quant  error
 *   ...   ...   ...    ...    ...
 * ```
 *
 * @param filename   the name of the file
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file
 * (default = 2)
 * @param indicators
 */
void RVData::load(const string filename, const string units, int skip,
                const vector<string>& indicators)
{
    data_t data;

    // Empty the vectors
    t.clear();
    y.clear();
    sig.clear();
    y2.clear();
    sig2.clear();

    // Read the file into the data container
    ifstream infile(filename);
    infile >> data;

    // Complain if something went wrong.
    if (!infile.eof()) {
        printf("Could not read data file (%s)!\n", filename.c_str());
        exit(1);
    }
    if (data.size() - skip == 0) {
        printf("Data file (%s) seems to be empty!\n", filename.c_str());
        exit(1);
    }

    if (skip < 0) skip = 0;

    // number of "columns" of data
    int ncol = data[skip].size();
    if (ncol < 3) {
        printf("Data file (%s) contains less than 3 columns!\n",
               filename.c_str());
        exit(1);
    }

    infile.close();

    datafile = filename;
    dataunits = units;
    dataskip = skip;
    datamulti = false;
    number_instruments = 1;

    double factor = 1.;
    if (units == "kms") factor = 1E3;

    for (size_t n = 0; n < data.size(); n++) {
        if (n < skip) continue;
        t.push_back(data[n][0]);
        y.push_back(data[n][1] * factor);
        sig.push_back(data[n][2] * factor);
        if (ncol > 3) {
            y2.push_back(data[n][3] * factor);
            sig2.push_back(data[n][4] * factor);
        }
    }

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];
    // Alternatively use the barycentre of Datapoints
    M0_epoch = std::accumulate(t.begin(),t.end(),0.0)/t.size();

#if VERBOSE
    // How many points did we read?
    printf("# Loaded %zu data points from file %s\n", t.size(),
           filename.c_str());

    // What are the units?
    if (units == "kms")
        printf("# Multiplied all RVs by 1000; units are now m/s.\n");
#endif
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
 * @param skip       number of lines to skip in the beginning of the file
 * (default = 2)
 */
void RVData::load_multi(const string filename, const string units, int skip)
{
    data_t data;

    // Empty the vectors
    t.clear();
    y.clear();
    sig.clear();
    y2.clear();
    sig2.clear();
    obsi.clear();

    // Read the file into the data container
    ifstream infile(filename);
    infile >> data;

    // Complain if something went wrong.
    if (!infile.eof()) {
        printf("Could not read data file (%s)!\n", filename.c_str());
        exit(1);
    }

    infile.close();

    datafile = filename;
    dataunits = units;
    dataskip = skip;
    datamulti = true;

    double factor = 1.;
    if (units == "kms") factor = 1E3;

    // the 4th column of the file identifies the instrument; can have "0"s
    // this is to make sure the obsi vector always starts at 1, to avoid
    // segmentation faults later
    vector<int> inst_id;
    int id_offset = 0;

    inst_id.push_back(data[skip][3]);

    for (size_t n = skip + 1; n < data.size(); n++) {
        if (data[n][3] != inst_id.back()) {
            inst_id.push_back(data[n][3]);
        }
    }

    id_offset = *min_element(inst_id.begin(), inst_id.end());

    for (unsigned n = skip; n < data.size(); n++) {
        // if (n < skip) continue;
        t.push_back(data[n][0]);
        y.push_back(data[n][1] * factor);
        sig.push_back(data[n][2] * factor);
        obsi.push_back(data[n][3] - id_offset + 1);
    }

    // How many points did we read?
    printf("# Loaded %zu data points from file %s\n", t.size(),
           filename.c_str());

    // Of how many instruments?
    set<int> s(obsi.begin(), obsi.end());
    printf("# RVs come from %zu different instruments.\n", s.size());
    number_instruments = s.size();

    if (units == "kms")
        cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];
    // Alternatively use the barycentre of Datapoints
    M0_epoch = std::accumulate(t.begin(),t.end(),0.0)/t.size();
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
 * @param skip       number of lines to skip in the beginning of the file
 * (default = 2)
 * @param indicators
 */
void RVData::load_multi(vector<string> filenames, const string units, int skip,
                      const vector<string>& indicators)
{
    data_t data;

    // Empty the vectors
    t.clear();
    y.clear();
    sig.clear();
    y2.clear();
    sig2.clear();
    obsi.clear();

    string dump;  // to dump the first skip lines of each file
    int filecount = 1;
    int last_file_size = 0;

    // Read the files into the data container
    for (auto& filename : filenames) {
        ifstream infile(filename);
        for (int i = 0; i < skip;
             i++)  // skip the first `skip` lines of each file
            getline(infile, dump);
        infile >> data;

        // Complain if something went wrong.
        if (!infile.eof()) {
            printf("Could not read data file (%s)!\n", filename.c_str());
            exit(1);
        }

        if (data.size() == 0) {
            printf("Data file (%s) seems to be empty!\n", filename.c_str());
            exit(1);
        }

        infile.close();

        // Assign instrument int identifier to obsi
        for (unsigned i = last_file_size; i < data.size(); i++) {
            obsi.push_back(filecount);
        }

        last_file_size = data.size();
        filecount++;
    }

    // number of "columns" of data
    int ncol = data[0].size();
    if (ncol < 3) {
        printf("Data files contain less than 3 columns!\n");
        exit(1);
    }

    datafile = "";
    datafiles = filenames;
    dataunits = units;
    dataskip = skip;
    datamulti = true;

    double factor = 1.;
    if (units == "kms") factor = 1E3;

    for (unsigned n = 0; n < data.size(); n++) {
        // if (n<skip) continue;
        t.push_back(data[n][0]);
        y.push_back(data[n][1] * factor);
        sig.push_back(data[n][2] * factor);
        if (ncol > 3) {
            y2.push_back(data[n][3] * factor);
            sig2.push_back(data[n][4] * factor);
        }
    }

    // How many points did we read?
    printf("# Loaded %zu data points from files\n", t.size());
    cout << "#   ";
    for (auto f : filenames) {
        cout << f.c_str();
        (f != filenames.back()) ? cout << " | " : cout << " ";
    }
    cout << endl;

    // Of how many instruments?
    set<int> s(obsi.begin(), obsi.end());
    // set<int>::iterator iter;
    // for(iter=s.begin(); iter!=s.end();++iter) {  cout << (*iter) << endl;}
    printf("# RVs come from %zu different instruments.\n", s.size());
    number_instruments = s.size();

    if (units == "kms")
        cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

    if (number_instruments > 1) {
        // We need to sort t because it comes from different instruments
        int N = t.size();
        vector<double> tt(N), yy(N), yy2(N);
        vector<double> sigsig(N), sig2sig2(N), obsiobsi(N);
        vector<int> order(N);

        // order = argsort(t)
        int x = 0;
        std::iota(order.begin(), order.end(), x++);
        sort(order.begin(), order.end(),
             [&](int i, int j) { return t[i] < t[j]; });

        for (unsigned i = 0; i < N; i++) {
            tt[i] = t[order[i]];
            yy[i] = y[order[i]];
            sigsig[i] = sig[order[i]];
            obsiobsi[i] = obsi[order[i]];
            if (ncol > 3) {
                yy2[i] = y2[order[i]];
                sig2sig2[i] = sig2[order[i]];
            }
        }

        for (unsigned i = 0; i < N; i++) {
            t[i] = tt[i];
            y[i] = yy[i];
            sig[i] = sigsig[i];
            obsi[i] = obsiobsi[i];
            if (ncol > 3) {
                y2[i] = yy2[i];
                sig2[i] = sig2sig2[i];
            }
        }
    }

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];
    // Alternatively use the barycentre of Datapoints
    M0_epoch = std::accumulate(t.begin(),t.end(),0.0)/t.size();
}

double RVData::get_RV_var() const
{
    double sum = accumulate(begin(y), end(y), 0.0);
    double mean = sum / y.size();

    double accum = 0.0;
    for_each(begin(y), end(y),
             [&](const double d) { accum += (d - mean) * (d - mean); });
    return accum / (y.size() - 1);
}

/**
 * @brief Calculate the maximum slope "allowed" by the data
 *
 * This calculates peak-to-peak(RV) / peak-to-peak(time), which is a good upper
 * bound for the linear slope of a given dataset. When there are multiple
 * instruments, the function returns the maximum of this peak-to-peak ratio of
 * all individual instruments.
 */
double RVData::topslope() const
{
    if (datamulti) {
        double slope = 0.0;
        for (size_t j = 0; j < number_instruments; j++) {
            vector<double> obsy, obst;
            for (size_t i = 0; i < y.size(); ++i) {
                if (obsi[i] == j + 1) {
                    obsy.push_back(y[i]);
                    obst.push_back(t[i]);
                }
            }
            const auto miny = min_element(obsy.begin(), obsy.end());
            const auto maxy = max_element(obsy.begin(), obsy.end());
            const auto mint = min_element(obst.begin(), obst.end());
            const auto maxt = max_element(obst.begin(), obst.end());
            double this_obs_topslope = (*maxy - *miny) / (*maxt - *mint);
            if (this_obs_topslope > slope) slope = this_obs_topslope;
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
double RVData::get_RV_span() const
{
    const auto min = min_element(y.begin(), y.end());
    const auto max = max_element(y.begin(), y.end());
    return *max - *min;
}

/**
 * @brief Calculate the maximum span (peak to peak) of the radial velocities
 *
 * This is different from get_RV_span only in the case of multiple instruments:
 * it returns the maximum of the spans of each instrument's RVs.
 */
double RVData::get_max_RV_span() const
{
    // for multiple instruments, calculate individual RV spans and return
    // largest
    if (datamulti) {
        double span = 0.0;
        for (size_t j = 0; j < number_instruments; j++) {
            vector<double> obsy;
            for (size_t i = 0; i < y.size(); ++i) {
                if (obsi[i] == j + 1) {
                    obsy.push_back(y[i]);
                }
            }
            const auto min = min_element(obsy.begin(), obsy.end());
            const auto max = max_element(obsy.begin(), obsy.end());
            double this_obs_span = *max - *min;
            if (this_obs_span > span) span = this_obs_span;
        }
        return span;
    }

    // for one instrument only, this is easy
    else {
        return get_RV_span();
    }
}

double RVData::get_adjusted_RV_var() const
{
    int ni;
    double sum, mean;
    vector<double> rva(t.size());

    for (size_t j = 0; j < number_instruments; j++) {
        ni = 0;
        sum = 0.;
        for (size_t i = 0; i < t.size(); i++)
            if (obsi[i] == j + 1) {
                sum += y[i];
                ni++;
            }
        mean = sum / ni;
        // cout << "sum: " << sum << endl;
        // cout << "mean: " << mean << endl;
        for (size_t i = 0; i < t.size(); i++)
            if (obsi[i] == j + 1) rva[i] = y[i] - mean;
    }

    mean = accumulate(rva.begin(), rva.end(), 0.0) / rva.size();
    double accum = 0.0;
    for_each(rva.begin(), rva.end(),
             [&](const double d) { accum += (d - mean) * (d - mean); });
    return accum / (y.size() - 1);
}

/**
 * @brief Order of magnitude of trend coefficient (of degree) given the data
 *
 * Returns the expected order of magnitude of the trend coefficient of degree
 * `degree` supported by the data. It calculates the order of magnitude of
 *    RVspan / timespan^degree
 */
int RVData::get_trend_magnitude(int degree) const
{
    return (int)round(log10(get_RV_span() / pow(get_timespan(), degree)));
}


/*******************************************************************************
 * 
 * LCData class
 * 
 ******************************************************************************/

LCData LCData::LCinstance;
LCData::LCData() {}


/**
 * @brief Load RV data from a file.
 *
 * Read a tab/space separated file with columns
 * ```
 *   time  vrad  error  quant  error
 *   ...   ...   ...    ...    ...
 * ```
 *
 * @param filename   the name of the file
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file
 * (default = 2)
 * @param indicators
 */
void LCData::load(const string filename, int skip)
{
    data_t data;

    // Empty the vectors
    t.clear();
    flux.clear();
    sig.clear();

    // Read the file into the data container
    ifstream infile(filename);
    infile >> data;

    // Complain if something went wrong.
    if (!infile.eof()) {
        printf("Could not read data file (%s)!\n", filename.c_str());
        exit(1);
    }
    if (data.size() - skip == 0) {
        printf("Data file (%s) seems to be empty!\n", filename.c_str());
        exit(1);
    }

    if (skip < 0) skip = 0;

    // number of "columns" of data
    int ncol = data[skip].size();
    if (ncol < 3) {
        printf("Data file (%s) contains less than 3 columns!\n",
               filename.c_str());
        exit(1);
    }

    infile.close();

    datafile = filename;
    dataskip = skip;

    double factor = 1.;
    // if (units == "kms") factor = 1E3;

    for (size_t n = 0; n < data.size(); n++) {
        if (n < skip) continue;
        t.push_back(data[n][0]);
        flux.push_back(data[n][1] * factor);
        sig.push_back(data[n][2] * factor);
    }

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];
    // Alternatively use the barycentre of Datapoints
    M0_epoch = std::accumulate(t.begin(),t.end(),0.0)/t.size();

#if VERBOSE
    // How many points did we read?
    printf("# Loaded %zu data points from file %s\n", t.size(),
           filename.c_str());
#endif
}

double LCData::get_flux_var() const
{
    double sum = accumulate(begin(flux), end(flux), 0.0);
    double mean = sum / flux.size();

    double accum = 0.0;
    for_each(begin(flux), end(flux),
             [&](const double d) { accum += (d - mean) * (d - mean); });
    return accum / (flux.size() - 1);
}

/**
 * @brief Calculate the maximum slope "allowed" by the data
 *
 * This calculates peak-to-peak(RV) / peak-to-peak(time), which is a good upper
 * bound for the linear slope of a given dataset. When there are multiple
 * instruments, the function returns the maximum of this peak-to-peak ratio of
 * all individual instruments.
 */
double LCData::topslope() const
{
    return get_flux_span() / get_timespan();
}

/**
 * @brief Calculate the total span (peak to peak) of the radial velocities
 */
double LCData::get_flux_span() const
{
    const auto min = min_element(flux.begin(), flux.end());
    const auto max = max_element(flux.begin(), flux.end());
    return *max - *min;
}


/**
 * @brief Order of magnitude of trend coefficient (of degree) given the data
 *
 * Returns the expected order of magnitude of the trend coefficient of degree
 * `degree` supported by the data. It calculates the order of magnitude of
 *    RVspan / timespan^degree
 */
int LCData::get_trend_magnitude(int degree) const
{
    return (int)round(log10(get_flux_span() / pow(get_timespan(), degree)));
}





/**
 * @brief Find pathnames matching a pattern
 *
 * from https://stackoverflow.com/a/8615450
 */
vector<string> glob(const string& pattern)
{
    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw runtime_error(ss.str());
    }

    // collect all the filenames into a vector<string>
    vector<string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    return filenames;
}