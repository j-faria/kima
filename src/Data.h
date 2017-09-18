#ifndef DNest4_Data
#define DNest4_Data

#include <vector>
#include <algorithm>

class Data
{
	private:
		std::vector<double> t, y, sig;
		std::vector<int> obsi;

	public:
		Data();
		void load(const char* filename);
		void loadnew(const char* filename, const char* fileunits);

		// Getters
		const std::vector<double>& get_t() const { return t; }
		double get_t_min() const { return *min_element(t.begin(), t.end()); }
		double get_t_max() const { return *max_element(t.begin(), t.end()); }

		const std::vector<double>& get_y() const { return y; }
		double get_y_min() const { return *min_element(y.begin(), y.end()); }
		double get_y_max() const { return *max_element(y.begin(), y.end()); }
		
		const std::vector<double>& get_sig() const { return sig; }
		const std::vector<int>& get_obsi() const { return obsi; }
		// double get_y_span() const {return abs(get_y_max() - get_y_min());}

	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};

#endif

