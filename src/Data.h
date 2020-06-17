// (c) 2019 João Faria
// This file is part of kima, which is licensed under the MIT license (see LICENSE for details)
#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <cmath>

class Data
{
	private:
		std::vector<double> t, y, sig;
		std::vector<int> obsi;
		std::vector<std::vector<double>> actind;

	public:
		Data();

		// to read data from one file, one instrument
		void load(const std::string filename, const std::string units, int skip=2, 
				  const std::vector<std::string>& indicators = std::vector< std::string >());

		// to read data from one file, more than one instrument
		void load_multi(const std::string filename, const std::string units, 
		                int skip=2);

		// to read data from more than one file, more than one instrument
		void load_multi(std::vector<std::string> filenames, const std::string units, int skip=2,
		                const std::vector<std::string>& indicators = std::vector< std::string >());


		bool indicator_correlations;
		int number_indicators;
		std::vector<std::string> indicator_names;

		std::string datafile;
		std::vector<std::string> datafiles;
		std::string dataunits;
		int dataskip;
		bool datamulti; // multiple instruments? not sure if needed
		int number_instruments;

		double M0_epoch;

		// to deprecate a function (C++14), put
		// [[deprecated("Replaced by bar, which has an improved interface")]]
		// before the definition

		/// @brief Get the number of RV points. @return int
		int N() const {return t.size();}

		/// @brief Get the array of times @return const std::vector<double>&
		const std::vector<double>& get_t() const { return t; }
		/// @brief Get the array of RVs @return const std::vector<double>&
		const std::vector<double>& get_y() const { return y; }
		/// @brief Get the array of errors @return const std::vector<double>&
		const std::vector<double>& get_sig() const { return sig; }

		/// @brief Get the mininum (starting) time @return double
		double get_t_min() const { return *std::min_element(t.begin(), t.end()); }
		/// @brief Get the maximum (ending) time @return double
		double get_t_max() const { return *std::max_element(t.begin(), t.end()); }
		/// @brief Get the timespan @return double
		double get_timespan() const { return get_t_max() - get_t_min(); }
		/// @brief Get the middle time @return double
		double get_t_middle() const { return get_t_min() + 0.5*get_timespan(); }

		/// @brief Get the mininum RV @return double
		double get_RV_min() const { return *std::min_element(y.begin(), y.end()); }
		/// @brief Get the maximum RV @return double
		double get_RV_max() const { return *std::max_element(y.begin(), y.end()); }
		/// @brief Get the RV span @return double
		double get_RV_span() const;
		/// @brief Get the maximum RV span @return double
		double get_max_RV_span() const;
		/// @brief Get the variance of the RVs @return double
		double get_RV_var() const;
		/// @brief Get the standard deviation of the RVs @return double
		double get_RV_std() const { return std::sqrt(get_RV_var()); }
		
		/// @brief Get the RV span, adjusted for multiple instruments @return double
		double get_adjusted_RV_span() const;
		/// @brief Get the RV variance, adjusted for multiple instruments @return double
		double get_adjusted_RV_var() const;
		/// @brief Get the RV standard deviation, adjusted for multiple instruments @return double
		double get_adjusted_RV_std() const { return std::sqrt(get_adjusted_RV_var()); }
		
		/// @brief Get the maximum slope allowed by the data. @return double
		double topslope() const;
		/// @brief Order of magnitude of trend coefficient (of degree) given the data
		int get_trend_magnitude(int degree) const;

		/// @brief Get the array of activity indictators @return std::vector<std::vector<double>>&
		const std::vector<std::vector<double>>& get_actind() const { return actind; }

		/// @brief Get the array of instrument identifiers @return std::vector<int>&
		const std::vector<int>& get_obsi() const { return obsi; }
		/// @brief Get the number of instruments. @return int
		int Ninstruments() const {std::set<int> s(obsi.begin(), obsi.end()); return s.size();}


	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};


template< class... Args >
void load( Args&&... args ) { Data::get_instance().load(args...); }

template< class... Args >
void load_multi( Args&&... args ) { Data::get_instance().load_multi(args...); }

