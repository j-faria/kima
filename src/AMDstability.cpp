#include "AMDstability.h"

namespace AMD
{
    /**
     * Returns the indices that would sort an array.
     * @param array input array
     * @return indices w.r.t sorted array
    */
    vector<size_t> argsort(const vd &array) {
        vector<size_t> indices(array.size());
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(),
                  [&array](int left, int right) -> bool {
                      // sort indices according to corresponding array element
                      return array[left] < array[right];
                  });
        return indices;
    }

    int AMD_stable(const vector<vd>& components, double star_mass)
    {
        //? order of parameters: P, K, φ, ecc, ω
        size_t NP = components.size();
        int is_stable = 0; // system is stable if is_stable=0
    
        vd periods(NP), masses(NP), sma(NP), eccentricities(NP), incs(NP);

        for (size_t i=0; i<NP; i++)
            periods[i] = components[i][0];

        auto indices = argsort(periods);
        
        for (auto i : indices)
        {
            double P = components[i][0];
            double K = components[i][1];
            double ecc = components[i][3];
            double m; // [Mjup]
            m = 4.919e-3 * pow(star_mass, 2./3) * pow(P, 1./3) * K * sqrt(1 - ecc*ecc);
            m = m * mjup2msun;  // [Msun]
            m = m / star_mass;  // planet/star mass ratio

            double a; // [AU]
            a = G13 * pow(star_mass, 1./3) * pow(P / (2 * M_PI), 2./3);

            // cout << "P: " << P << "\t" << "K: " << K << '\t';
            // cout << "e: " << ecc << '\t' << "mass: " << m;
            // cout << endl;

            masses[i] = m;
            sma[i] = a;
            eccentricities[i] = ecc;
            incs[i] = 0.0;
        }

        auto Lambda = am_circular(masses, sma);
        double AMD = total_AMD_system(masses, sma, eccentricities, incs);

        // Can planet 1 collide with the star?
        if (AMD > Lambda[0])
            return COLLISION;

        for (size_t k = 1; k < NP; k++)
        {
            double Cx = AMD / Lambda[k];
            is_stable = AMD_stability_pair(masses[k-1], masses[k], 
                                           sma[k-1], sma[k], Cx);
        }

        return is_stable;
    }

    /*
        Compute the total AMD of a system.
        @param μ : list of planet/star mass ratios
        @param a : list of semimajor axes
        @param e : list of eccentricities
        @param i : list of mutual inclinations (in radians)
            Note: μ and a must be sorted from innermost to outermost planet
        @return AMD : the total AMD of the system.
    */
    double total_AMD_system(vd mu, vd a, vd e, vd i)
    {
        auto Lambda = am_circular(mu, a);
        double AMD = 0.0;
        for (size_t k=0; k<mu.size(); k++)
            AMD += Lambda[k] * (1 - sqrt((1 - e[k]) * (1 + e[k])) * cos(i[k]));
        return AMD;
    }

    vd am_circular(vd mu, vd a)
    {
        vd amc(mu.size());
        for (size_t i = 0; i < mu.size(); i++)
            amc[i] = mu[i] * sqrt(a[i]);
        return amc;
    }


    /*
        Determines the AMD stability of a pair of planets using the collision
        condition and the MMR overlap conditions of Laskar & Petit (2017) and 
        Petit & Laskar (2017).

        @param μ1 Planet/star mass ratio of inner planet
        @param μ2 Planet/star mass ratio of outer planet
        @param a1 Semimajor axis of inner planet
        @param a2 Semimajor axis of outer planet
        @param Cx Relative AMD (Eq. 29 of Laskar & Petit 2017)

        @return stable
                STABLE          AMD stable
                COLLISION       Fails collision condition
                MMR_CIRCULAR    Fails MMR overlap condition for circular orbits
                MMR_ECCENTRIC   Fails MMR overlap condition for eccentric orbits
    */
    int AMD_stability_pair(double mu1, double mu2, double a1, double a2, double Cx)
    {
        double y = mu1 / mu2;
        double a = a1 / a2;
        double e = mu1 + mu2; // Equation 4 of Petit, Laskar, & Boue 2017

        // Equation 76 and Section 3.6 of Petit, Laskar, & Boue 2017
        double a_crit = 1 - 1.46 * pow(e, 2./7);
        if (a > a_crit)  // fails MMR circular
            return MMR_CIRCULAR;

        // Relative AMD from collision condition (Laskar & Petit 2017)
        double C_coll = relative_AMD_collision(mu1, mu2, a1, a2);

        // Relative AMD from MMR overlap condition (Petit, Laskar, & Boue 2017)
        double C_mmr = relative_AMD_MMR_overlap(mu1, mu2, a1, a2);

        // Final result
        double C_crit = min(C_coll, C_mmr);
        double ratio = Cx / C_crit;
        if (Cx < C_crit)
            return STABLE;
        else if (C_coll < C_mmr)
            return COLLISION;
        else
            return MMR_ECCENTRIC;
    }





    /*
        # Compute the critical (minimum) relative AMD for collision, based on Equations 29 & 39 in Laskar & Petit (2017).
        # # Arguments:
        # - `μ1`: planet/star mass ratio of inner planet.
        # - `μ2`: planet/star mass ratio of outer planet.
        # - `a1`: semimajor axis of inner planet.
        # - `a2`: semimajor axis of outer planet.
        # # Returns:
        # - `C_coll::Float64`: critical relative AMD for collision.
    */
    double relative_AMD_collision(double mu1, double mu2, double a1, double a2)
    {
        double y = mu1 / mu2;
        double a = a1 / a2;

        double e1 = critical_eccentricity(y, a);
        double e2 = 1 - a - a * e1;
        double C_coll = y * sqrt(a) * (1 - sqrt((1 - e1) * (1 + e1))) + (1 - sqrt((1 - e2) * (1 + e2)));
        return C_coll;
    }


    /*
        # Finds the root of F(e,γ,α) from Equation 35 of Laskar & Petit (2017).
        # INPUT:
        #         γ		m_in / m_out ; planet-planet mass ratio.
        #         α		a_in / a_out ; semimajor axis ratio
        #         [tol]	Tolenrance (default: 1e-15)
        # OUTPUT: Critical eccentricity of the inner planet.
    */
    double critical_eccentricity(double y, double a)
    {
        double tol = 1e-15;

        // Find the root of F() by Newton's method + bisection
        double eR, FR, eL, FL, ex;
        double e0 = 0.5; // Initial guess
        double F0 = F(e0, y, a);

        if (F0 >= 0)
        {
            eR = e0;
            FR = F0;
            eL = 0.0;
            ex = 0.0;
            FL = F(eL, y, a);
        }
        else 
        {
            eL = e0;
            FL = F0;
            eR = 1.0;
            ex = 1.0;
            FR = F(eR, y, a);
        }

        while (abs(ex - e0) > tol)
        {
            // Try a Newton step first
            ex = e0 - F0 / dFde(e0, y, a);
            if (ex <= eL | ex >= eR)
            {
                // Newton's method jumped out of the interval.
                // Switch to linear interpolation.
                double m = (FR - FL) / (eR - eL);
                double b = F0 - m * e0;
                ex = -b / m;
            }

            // Swap and update
            swap(ex, e0);
            F0 = F(e0, y, a);

            if (F0 >= 0)
            {
                eR = e0;
                FR = F0;
            }
            else
            {
                eL = e0;
                FL = F0;
            }
        }
        return ex;
    }


    // F(e,γ,α) from Equation 35 of Laskar & Petit (2017)
    double F(double e, double y, double a)
    {
        return a * e + y * e / sqrt(a * (1 - e) * (1 + e) + y*y * e*e) - 1.0 + a;
    }

    // dF/de from Equation 36 of Laskar & Petit (2017). dF/de > 0 always
    double dFde(double e, double y, double a)
    {
        return a + a * y / pow(a * (1 - e) * (1 + e) + y*y * e*e, 1.5);
    }

    /*
        # relative_AMD_MMR_overlap(μ1, μ2, a1, a2)
        # Compute the critical (minimum) relative AMD for MMR overlap, based on Equation 74 in Petit, Laskar, & Boue (2017).
        # # Arguments:
        # - `μ1`: planet/star mass ratio of inner planet.
        # - `μ2`: planet/star mass ratio of outer planet.
        # - `a1`: semimajor axis of inner planet.
        # - `a2`: semimajor axis of outer planet.
        # # Returns:
        # - `C_mmr::Float64`: critical relative AMD for MMR overlap.
    */
    double relative_AMD_MMR_overlap(double mu1, double mu2, double a1, double a2)
    {
        double y = mu1 / mu2;
        double a = a1 / a2;
        double e = mu1 + mu2;
        double r = 0.80199; // Equation 28 of Petit, Laskar, & Boue (2017)

        double g = ( (81 * pow(1 - a, 5)) / (512 * r * e)) - ((32 * r * e) / (9 * pow(1 - a, 2)));
        double C_mmr = (g*g * y * sqrt(a)) / (2 + 2 * y * sqrt(a));
        return C_mmr;
    }


}