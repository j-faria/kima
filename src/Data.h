#ifndef DNest4_Data
#define DNest4_Data

#include <vector>
#include <algorithm>

class Data
{
	private:
		std::vector<double> t, y, sig;

	public:
		Data();
		//void load(const char* filename);
		void load(const char* filename, const char* units, int skip=2);
		int index_fibers;

		// Getters
		int N() const {return t.size();}
		const std::vector<double>& get_t() const { return t; }
		double get_t_min() const { return *min_element(t.begin(), t.end()); }
		double get_t_max() const { return *max_element(t.begin(), t.end()); }

		const std::vector<double>& get_y() const { return y; }
		double get_y_min() const { return *min_element(y.begin(), y.end()); }
		double get_y_max() const { return *max_element(y.begin(), y.end()); }
		double get_RV_span() const {return get_y_max() - get_y_min();}
		
		const std::vector<double>& get_sig() const { return sig; }
		double topslope() const {return abs(get_y_max() - get_y_min()) / (t.back() - t.front()); }

	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};

#endif

