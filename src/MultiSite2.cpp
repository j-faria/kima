#include "MultiSite2.h"

using namespace std;

namespace dataset
{

class DataSet : public Multiton<string, DataSet> 
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
/*            if (this->id == 0)
                cout<<"# Full dataset: "<<t.size()<<" points from file "<<filename<<endl;
            else
*/                cout<<"# - dataset: "<<t.size()<<" points from file "<<filename<<endl;

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


}

//Foo* foo2 = Foo::getPtr("foobar");
//Foo::destroy();

