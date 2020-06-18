// This file contains all the options that can be found in kima_setup.cpp files. 
// Note that this file is for demonstration, it *is not* valid C++, and so it 
// will not compile. On the other hand, the kima_setup.cpp files *must* contain 
// valid C++ code.


// include some kima defaults and definitions
#include "kima.h"

// sometimes it's also useful to do (e.g. for printing values to the terminal)
using namespace std;


// whether the model includes a Gaussian process component
const bool GP = false;

// whether the model includes a Moving Average component
const bool MA = false;

// whether the model includes a trend, of what degree
// (for a linear trend, degree=1; up to degree=3)
const bool trend = false;
const int degree = 0;

// whether the data comes from different instruments
// (and offsets + individual jitters should be included in the model)
const bool multi_instrument = false;

// include (better) known extra Keplerian curve(s)? (KO mode!)
// this requires setting extra priors for the orbital parameters (see below)
const bool known_object = false;
const int n_known_object = 0;  // how many?

// use a Student-t distribution for the likelihood (instead of a Gaussian)
const bool studentt = false;

// whether the model includes hyper-priors for the orbital period and semi-amplitude
const bool hyperpriors = false;


// this is the RVmodel constructor, where the model is built
RVmodel::RVmodel():
    fix(true), // fix the number of planets? (to npmax)
    npmax(1)   // maximum number of planets in the model (or fixed number of planets if `fix` is true)
{
    /* the data object */
    /*******************/

    // the data "object" can accessed here by using
    auto data = get_data();

    // and it has a number of methods, accessible with data.method(), whose 
    // output can be used when setting priors, for example

		// get the number of RV points
		N()

		// get the array of times
		get_t()
		// get the array of RVs
		get_y()
		// get the array of errors
		get_sig()

		// get the mininum (starting) time
		get_t_min()
		// get the maximum (ending) time
		get_t_max()
		// get the timespan
		get_timespan()
		// get the middle time (t_min + 0.5*timespan)
		get_t_middle()

		// get the mininum RV
		get_y_min()
		// get the maximum RV
		get_y_max()
		// get the RV span
		get_RV_span()
		// get the variance of the RVs
		get_RV_var()
		// get the standard deviation of the RVs
		get_RV_std()
		
		// get the RV span, adjusted for multiple instruments
		get_adjusted_RV_span()
		// get the RV variance, adjusted for multiple instruments
		get_adjusted_RV_var()
		// get the RV standard deviation, adjusted for multiple instruments
		get_adjusted_RV_std()
		
		// get the maximum slope allowed by the data
		topslope()

		// get the array of activity indictators
		get_actind()

		// get the array of instrument identifiers
		get_obsi()
		
        // get the number of instruments.
		Ninstruments()


    /* priors */
    /**********/

    // see more documentation about setting the priors at 
    // https://github.com/j-faria/kima/wiki/Changing-the-priors
    // 
    // note: LogUniform is usually called the Jeffreys prior
    // note: ModifiedLogUniform is usually called the modified Jeffreys prior

   
    // set the priors for the orbital period(s) and semi-amplitude(s)

    // if hyperpriors=false (default)
    Pprior = make_shared<LogUniform>(1.0, 1e5);
    Kprior = make_shared<ModifiedLogUniform>(1.0, 1e3);


    // eccentricity(ies)
    eprior = make_shared<Uniform>(0, 1); // default
    eprior = make_shared<Kumaraswamy>(0.867, 3.03); // another typical option, 
                                                    // approximates the Beta 
                                                    // prior first proposed by
                                                    // Kipping (2013)

    // orbital phase(s) 
    // note: this corresponds directly to the mean anomaly at the epoch
    phiprior = make_shared<Uniform>(0, 2*PI);

    // longitude(s) of periastron
    wprior = make_shared<Uniform>(0, 2*PI);


    // systemic velocity, always in m/s
    Cprior = make_prior<Uniform>(data.get_RV_min(), data.get_RV_max()); // default

    // additional white noise, always in m/s
    // note: if multi_instrument=true, all jitters share this prior
    Jprior = make_prior<ModifiedLogUniform>(1.0, 100.);

    // slope of the linear trend (if trend=true), always in m/s/day
    slope_prior = make_prior<Uniform>( -data.topslope(), data.topslope() );

    // between-instrument offsets, always in m/s
    offsets_prior = make_prior<Uniform>( -data.get_RV_span(), data.get_RV_span() );


    /* GP hyperparameters */
    // note: for eta1 and eta4 the priors are in the log of the parameter
    log_eta1_prior = make_prior<Uniform>(-5, 5);
    
    // note: eta2 can correspond to the active region evolution timescale; may 
    // want to change this prior according to the star
    eta2_prior = make_prior<LogUniform>(1, 100);

    // note: eta3 can correspond to the stellar rotation period; you probably 
    // want to change this prior according to the star
    eta3_prior = make_prior<Uniform>(10, 40);

    // note: this default prior for eta4 is tipically a very good choice
    log_eta4_prior = make_prior<Uniform>(-1, 1);



    /* moving average parameters */
    sigmaMA_prior = make_prior<ModifiedLogUniform>(1.0, 10.);
    tauMA_prior = make_prior<LogUniform>(1, 10);


    // correlation coefficients with activity indicators
    betaprior = make_prior<Gaussian>(0, 1);


    /* if hyperpriors=true */
    // note: these are *very* broad priors

    // hyperparameters for orbital period and semi-amplitude hierachical priors
    log_muP_prior = make_shared<TruncatedCauchy>(log(365), 1., log(365)-21, log(365)+21);
    wP_prior = make_shared<Uniform>(0.1, 3);
    log_muK_prior = make_shared<TruncatedCauchy>(0., 1., 0.-21, 0.+21);

    Pprior = make_shared<Laplace>(exp(log_muP), wP);
    Kprior = make_shared<Exponential>(exp(log_muK));


}



int main(int argc, char** argv)
{
    /* 
    see more documentation about the input data at 
    http://github.com/j-faria/kima/wiki/Input-data 
    */
 

    /******************************************************/
    /* when reading data of one instruments from one file */

    // set the RV data file
    datafile = "filename";

    // Load the data. RV units can be "ms" for m/s or "kms" for km/s. The third 
    // argument sets the number of lines to skip in the file's header.
    load(datafile, "kms", 0);


    /******************************************************************/
    /* when reading data of multiple instruments from different files */
    
    // set the RV data files from multiple instruments
    datafiles = {"filename_1", "filename_2", "filename_3"};

    // Load the data. RV units can be "ms" for m/s or "kms" for km/s. The third 
    // argument sets the number of lines to skip in the files' headers.
    // note: all files should have the same structure in terms of the three 
    // first columns, and should all be in the same units
    load_multi(datafiles, "ms", 2);


    /******************************************************************/
    /* when reading data of multiple instruments from one single file */

    // set the RV data file
    datafile = "filename_joined";

    // Load the data. RV units can be "ms" for m/s or "kms" for km/s. The third 
    // argument sets the number of lines to skip in the files' headers.
    // note: the file should contain a fourth column with an instrument ID
    load_multi(datafile, "ms", 1);



    // create the DNest4 sampler (never need to change this line)
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);

    // run the sampler
    sampler.run(1);


    // return from main (never need to change this line)
    return 0;
}
