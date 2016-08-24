#ifndef DNest4_MultiSite
#define DNest4_MultiSite

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

// the maximum number of individual datasets!
const int nton_limit = 10;

template<typename Derived, int n = 1>
class nton_base // Singletons are the default
{
    protected:

        // Prevent nton_base to be used outside of inheritance hierarchies
        nton_base() {} 

        // Prevent nton_base (and Derived classes) from being copied
        nton_base(const nton_base&) = delete;


    public:
        int id = 0;
        // only the 0th instance should set and get this
        int nsites = 0;

        int N;

        // Get first element by default, useful for Singletons
        static Derived& get_instance(int i = 0)
        {
            assert(i < nton_limit && "I cannot hold so many datasets. Look in MultiSite.h!");
            static Derived instance;

            instance.id = i; // set id of this instance

            return instance;
        }

        void set_nsites(int ns)
        {
            static Derived instance;
            nsites = ns;
        }

};


/*
A class that can hold up to 10 (nton_limit) RV datasets, each with its own
time series of times, rvs and errors.
The 'singleton' (the 0th instance) should hold the complete dataset.
*/
class MultiSite : public nton_base<MultiSite, nton_limit>
{
    private:
        std::vector<double> t, y, sig;

    public:
        // RV offset, always relative to the background vsys
        double offset;

        void idme()
        {
            std::cout << "I'm object " << this << "\n";
        }

        void load(const char* filename, const char* units = "kms")
        {
            //std::cout << this->id << "\n";

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
            if (this->id == 0)
                cout<<"# Full dataset: "<<t.size()<<" points from file "<<filename<<endl;
            else
                cout<<"# - dataset "<<this->id<<": "<<t.size()<<" points from file "<<filename<<endl;

            fin.close();
            //cout<<it<<endl;

            this->N = t.size();

            if (units == "kms"){
                double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
                //cout<<mean<<endl;

                // this is probably a stupid way to substract the mean and convert to m/s
                std::transform( y.begin(), y.end(), y.begin(), std::bind2nd( minus<double>(), mean ) );
                std::transform( y.begin(), y.end(), y.begin(), std::bind2nd( multiplies<double>(), 1000. ) );
                std::transform( y.begin(), y.end(), y.begin(), std::bind2nd( plus<double>(), mean ) );

                // the errorbars just need to be converted to m/s
                std::transform( sig.begin(), sig.end(), sig.begin(), std::bind2nd( multiplies<double>(), 1000. ) );
            }
            
            //for (std::vector<double>::const_iterator i = sig.begin(); i != sig.end(); ++i)
            //std::cout << *i << '\n';
            //std::cout << '\n';
        }

        // Getters
        const std::vector<double>& get_t() const { return t; }
        const std::vector<double>& get_y() const { return y; }
        const std::vector<double>& get_sig() const { return sig; }
        double get_y_min() const { return *min_element(y.begin(), y.end()); }
        double get_y_max() const { return *max_element(y.begin(), y.end()); }
        double get_tspan() const { return *max_element(t.begin(), t.end()) - *min_element(t.begin(), t.end());}

};


#endif