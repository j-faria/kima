#pragma once


namespace kima {

    // this creates an alias for std::make_shared
    /**
     * @brief Assign a prior distribution.
     * 
     * This function defines, initializes, and assigns a prior distribution.
     * Possible distributions are ...
     * 
     * For example:
     * 
     * @code{.cpp}
     *          Cprior = make_prior<Uniform>(0, 1);
     * @endcode
     * 
     * @tparam T     ContinuousDistribution
     * @param args   Arguments for constructor of distribution
    */
    template <class T, class... Args>
    std::shared_ptr<T> make_prior(Args&&... args)
    {
        return std::make_shared<T>(args...);
    }

}