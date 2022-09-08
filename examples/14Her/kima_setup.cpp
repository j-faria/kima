#include "kima.h"

const bool MA = false;
const bool multi_instrument = false;
const bool known_object = false;
const int n_known_object = 0;
const bool studentt = false;

RVmodel::RVmodel():fix(true),npmax(0)
{
}


int main(int argc, char** argv)
{
    /* set the RV data file */
    datafile = "14her.rdb";
    // datafile = "test.rdb";

    // the third (optional) argument,
    // tells kima not to skip any line in the header of the file
    load(datafile, "ms", 2);

    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
