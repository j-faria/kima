#include "kima.h"


RVmodel::RVmodel():fix(true),npmax(1)
{
    // set priors and model options here
}


int main(int argc, char** argv)
{
    // set one of the `datafile` or `datafiles` variables and use the load() or
    // load_multi() functions to read them
    // 
    // examples:
    // 
    // datafile = "data.txt";
    // load(datafile, "ms", 0);
    // 
    // or
    // 
    // datafiles = {"data_file_1", "data_file_2"};
    // load_multi(datafiles, "ms", 0);


    /// set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);
}
