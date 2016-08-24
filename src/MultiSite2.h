#ifndef MULTITON_H
#define MULTITON_H

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>

using namespace std;

template <typename Key, typename T> class Multiton
{
public:    
    
    static std::map<Key, T*> instances;

    static void destroy()
    {
        for (typename std::map<Key, T*>::iterator it = instances.begin(); it != instances.end(); ++it) {
            delete (*it).second;
        }
    }

    static T& getRef(const Key& key)
    {
        typename std::map<Key, T*>::iterator it = instances.find(key);

        if (it != instances.end()) {
            return *(T*)(it->second);
        }

        T* instance = new T;
        instances[key] = instance;
        return *instance;
    }

    static T* getPtr(const Key& key)
    {
        typename std::map<Key, T*>::iterator it = instances.find(key);

        if (it != instances.end()) {
            return (T*)(it->second);
        }

        T* instance = new T;
        instances[key] = instance;
        return instance;
    }


    int nsites() {return instances.size();}
    int Ntotal()
    {
        int Nt = 0;
        for (auto iterator : instances)
            Nt += iterator.second->N;
        return Nt;
    }

    int N;
    int id;


protected:
    Multiton() {}
    virtual ~Multiton() {}

private:
    Multiton(const Multiton&) {}
    Multiton& operator= (const Multiton&) { return *this; }

};

template <typename Key, typename T> std::map<Key, T*> Multiton<Key, T>::instances;


class DataSet : public Multiton<string, DataSet> 
{
    private:
        std::vector<double> t, y, sig;

    public:
        std::vector<double> observatory_id;
        // RV offset, always relative to the background vsys
        double offset;

        void idme()
        {
            std::cout << "I'm object " << this << "\n";
            //std::cout << "there are " << this->instances.size() << " like me" << "\n";
        }

        void load(const char* filename, const char* units = "kms", int id = 0)
        {

            fstream fin(filename, ios::in);
            if(!fin)
            {
                cerr<<"# Error. Couldn't open file "<<filename<<endl;
                exit(1);
            }

            // Empty the vectors
            t.clear();
            y.clear();
            sig.clear();
            observatory_id.clear();

            int it = 0;
            double temp1, temp2, temp3;
            while(fin>>temp1 && fin>>temp2 && fin>>temp3)
            {
                t.push_back(temp1);
                y.push_back(temp2);
                sig.push_back(temp3);
                observatory_id.push_back(id);
                it++;
            }
            
            //std::fill(observatory_id.begin(), observatory_id.begin()+t.size(), id);
            //cout<<observatory_id.size();

            cout<<"# Read "<<t.size()<<" points from file "<<filename<<", dataset "<< id <<endl;

            fin.close();

            this->N = t.size();
            this->id = id;

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


        // Get the complete data of all loaded datasets
        void fill_full_y(std::vector<double>& in) const 
        {
            for (auto iterator : this->instances) {
                const vector<double>& yi = iterator.second->get_y();
                in.insert(std::end(in), std::begin(yi), std::end(yi));
            }
        }


};


#endif