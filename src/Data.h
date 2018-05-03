#ifndef DNest4_Data
#define DNest4_Data

#include <vector>
#include <algorithm>
#include <cmath>

class Data
{
	private:
		std::vector<double> t, y, sig;

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

		const std::vector<double>& get_t() const { return t; }
		double get_t_min() const { return *std::min_element(t.begin(), t.end()); }
		double get_t_max() const { return *std::max_element(t.begin(), t.end()); }
		double get_t_middle() const { return get_t_min() + 0.5*(get_t_max() - get_t_min()); }

		const std::vector<double>& get_y() const { return y; }
		double get_y_min() const { return *std::min_element(y.begin(), y.end()); }
		double get_y_max() const { return *std::max_element(y.begin(), y.end()); }
		double get_RV_span() const { return get_y_max() - get_y_min(); }
		double get_RV_var() const;
		double get_RV_std() const { return std::sqrt(get_RV_var()); }
		
		const std::vector<double>& get_sig() const { return sig; }
		double topslope() const {return std::abs(get_y_max() - get_y_min()) / (t.back() - t.front());}

	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};

#endif

