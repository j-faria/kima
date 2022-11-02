#pragma once

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <numeric>
#include <optional>
#include <vector>
#include <map>

using namespace std;

struct loadtxt {
    // Mandatory arguments
    loadtxt(string fname) : _fname(fname) {}

    // Optional arguments
    // ('this' is returned for chaining)
    loadtxt& comments(string comments) { _comments = comments; return *this; }
    loadtxt& delimiter(string delimiter) { _delimiter = delimiter; return *this; }
    loadtxt& skiprows(int skiprows) { _skiprows = skiprows; return *this; }
    loadtxt& usecols(vector<int> usecols) { _usecols = usecols; return *this; }
    loadtxt& max_rows(int max_rows) { _max_rows = max_rows; return *this; }

    vector<vector<double>> operator()()
    {

        ifstream infile(_fname);

        if (!infile.good()) {
            cout << "Could not read file (" << _fname << ")!\n";
            exit(1);
        }

        // ignore first `skiprows` lines
        static const int max_line = 65536;
        for (int i = 0 ; i < _skiprows ; i++)
            infile.ignore(max_line, '\n');

        vector<double> record;

        while (true)
        {
            // clear the record before reading
            record.clear();

            // read the entire line into a string (a record is terminated by a newline)
            string line;
            getline(infile, line);

            if (infile.eof() && line.empty()) {
                break;
            }
            
            if (line.find(_comments, 0) == 0)
                continue;

            // use a stringstream to separate the fields out of the line
            stringstream ss(line);

            // convert each field to a double
            // and add the newly-converted field to the end of the record

            if (_delimiter == " ") {
                double f;
                while (ss >> f)
                    record.push_back(f);
            }
            else {
                string val;
                while (getline(ss, val, _delimiter[0]))
                    record.push_back(stod(val));
            }


            _filedata.push_back(record);

        }
        
        // complain if something went wrong
        if (!infile.eof())
        {
            cout << "Could not read file (" << _fname << ")!\n";
            exit(1);
        }

        infile.close();

        int nlines = _filedata.size();

        if (nlines <= 0)
        {
            cout << "File seems to be empty (" << _fname << ")!\n";
            exit(1);
        }

        int ncols = _filedata[0].size();

        vector<int> cols;
        if (_usecols.size() == 0)
        {
            data.resize(ncols);
            cols.resize(ncols);
            std::iota(cols.begin(), cols.end(), 0);
        }
        else
        {
            data.resize(_usecols.size());
            cols = _usecols;
            //! usecols starts from 1!
            for (auto& j : cols)
                j--;
        }

        for (size_t i = 0; i < data.size(); i++)
            data[i].reserve(nlines);
        
        for (int i = 0; i < nlines; i++)
        {
            int k = 0;
            for (auto j : cols)
            {
                data[k].push_back(_filedata[i][j]);
                k++;
            }
        }
        
        return data;
    }

    ~loadtxt(){};

    public:
        vector<vector<double>> data;

    private:
        string _fname;
        string _comments = "#";
        string _delimiter = " ";
        int _skiprows = 0;
        vector<int> _usecols;
        int _max_rows;
        vector<vector<double>> _filedata;
};


struct loadrdb : loadtxt {
    loadrdb(string fname) : loadtxt(fname), _fname(fname) { skiprows(2); }

    private:
        string _fname;
};




    /*
        Load data from a text file.

        Each row in the text file must have the same number of values.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            File, filename, or generator to read.  If the filename extension is
            ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings.
        dtype : data-type, optional
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        comments : str or sequence of str, optional
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter : str, optional
            The string used to separate values. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will parse the
            column string into the desired value.  E.g., if column 0 is a date
            string: ``converters = {0: datestr2num}``.  Converters can also be
            used to provide a default value for missing data (but see also
            `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines, including comments; default: 0.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.

            .. versionchanged:: 1.11.0
                When a single column has to be read it is possible to use
                an integer instead of a tuple. E.g ``usecols = 3`` reads the
                fourth column the same way as ``usecols = (3,)`` would.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
            data-type, arrays are returned for each field.  Default is False.
        ndmin : int, optional
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed.
            Legal values: 0 (default), 1 or 2.

            .. versionadded:: 1.6.0
        encoding : str, optional
            Encoding used to decode the inputfile. Does not apply to input streams.
            The special value 'bytes' enables backward compatibility workarounds
            that ensures you receive byte arrays as results if possible and passes
            'latin1' encoded strings to converters. Override this value to receive
            unicode arrays and pass strings as input to converters.  If set to None
            the system default is used. The default value is 'bytes'.

            .. versionadded:: 1.14.0
        max_rows : int, optional
            Read `max_rows` lines of content after `skiprows` lines. The default
            is to read all the lines.

            .. versionadded:: 1.16.0
    */