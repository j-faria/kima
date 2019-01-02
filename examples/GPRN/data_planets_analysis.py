##### Defining the important stuff #####
#address to posterior_sample.txt
address = "posterior_sample.txt"

#number of planets found
NumberOfPlanets = 1
#number of planets defined in the RVmodel constructor
ConstructorNumber = 2

##### Reading the data #####
import numpy as np
data=np.loadtxt(address)

##### Selection of posterior number of the planet #####
nonplanet = np.where(data[:,7]!= NumberOfPlanets)
new_data = np.delete(data,nonplanet,0)

##### GP #####
s = np.percentile(new_data[:,0],50)
s_up = np.percentile(new_data[:,0],84)-np.percentile(new_data[:,0],50)
s_low = np.percentile(new_data[:,0],50)-np.percentile(new_data[:,0],16)
print('s = {0} +{1} -{2}'.format(s,s_up,s_low))     

eta1 = np.percentile(new_data[:,1],50)
eta1_up = np.percentile(new_data[:,1],84)-np.percentile(new_data[:,1],50)
eta1_low = np.percentile(new_data[:,1],50)-np.percentile(new_data[:,1],16)
print('eta1 = {0} +{1} -{2}'.format(eta1,eta1_up,eta1_low))     


eta2 = np.percentile(new_data[:,2],50)
eta2_up = np.percentile(new_data[:,2],84)-np.percentile(new_data[:,2],50)
eta2_low = np.percentile(new_data[:,2],50)-np.percentile(new_data[:,2],16)
print('eta2 (days) = {0} +{1} -{2}'.format(eta2,eta2_up,eta2_low))   

eta3 = np.percentile(new_data[:,3],50)
eta3_up = np.percentile(new_data[:,3],84)-np.percentile(new_data[:,3],50)
eta3_low = np.percentile(new_data[:,3],50)-np.percentile(new_data[:,3],16)
print('eta3 (days) = {0} +{1} -{2}'.format(eta3,eta3_up,eta3_low))

eta4 = np.percentile(new_data[:,4],50)
eta4_up = np.percentile(new_data[:,4],84)-np.percentile(new_data[:,4],50)
eta4_low = np.percentile(new_data[:,4],50)-np.percentile(new_data[:,4],16)
print('eta4 = {0} +{1} -{2}'.format(eta4,eta4_up,eta4_low))
print()

##### Planets analysis #####
P=8	#period -> column were periods start
K=P+ConstructorNumber #semi-amplitude
PHI=K+ConstructorNumber #phi
ECC=PHI+ConstructorNumber #eccentricity
OMEGA=ECC+ConstructorNumber #velocity

##to limit the "range" of the period we are counting
#new_data1 = np.zeros_like(data[1,:])
#for j in range(data[:,0].size):
#    if data[j,8] < 100:
#        new_data1 = np.vstack((new_data1,new_data[j,:]))
#
#new_data = new_data1[1:,:]
for i in range(ConstructorNumber):
    pl_P = np.percentile(new_data[:,P+i],50)
    if pl_P != 0:
        pl_Pup = np.percentile(new_data[:,P+i],84)-np.percentile(new_data[:,P+i],50)
        pl_Plow = np.percentile(new_data[:,P+i],50)-np.percentile(new_data[:,P+i],16)
        print('Period (days)= {0} +{1} -{2}'.format(pl_P,pl_Pup,pl_Plow)) 

        pl_K = np.percentile(new_data[:,K+i],50)
        pl_Kup = np.percentile(new_data[:,K+i],84)-np.percentile(new_data[:,K+i],50)
        pl_Klow = np.percentile(new_data[:,K+i],50)-np.percentile(new_data[:,K+i],16)
        print('K (m/s) = {0} +{1} -{2}'.format(pl_K,pl_Kup,pl_Klow)) 

        pl_phi = np.percentile(new_data[:,PHI+i],50)
        pl_phiup = np.percentile(new_data[:,PHI+i],84)-np.percentile(new_data[:,PHI+i],50)
        pl_philow = np.percentile(new_data[:,PHI+i],50)-np.percentile(new_data[:,PHI+i],16)
        print('phi = {0} +{1} -{2}'.format(pl_phi,pl_phiup,pl_philow)) 

        pl_E = np.percentile(new_data[:,ECC+i],50)
        pl_Eup = np.percentile(new_data[:,ECC+i],84)-np.percentile(new_data[:,ECC+i],50)
        pl_Elow = np.percentile(new_data[:,ECC+i],50)-np.percentile(new_data[:,ECC+i],16)
        print('Ecc = {0} +{1} -{2}'.format(pl_E,pl_Eup,pl_Elow)) 

        pl_OMEGA = np.percentile(new_data[:,OMEGA+i],50)
        pl_OMEGAup = np.percentile(new_data[:,OMEGA+i],84)-np.percentile(new_data[:,OMEGA+i],50)
        pl_OMEGAlow = np.percentile(new_data[:,OMEGA+i],50)-np.percentile(new_data[:,OMEGA+i],16)
        print('Omega = {0} +{1} -{2}'.format(pl_OMEGA,pl_OMEGAup,pl_OMEGAlow))

        #dividing for 1000 because of m/s
        mass = new_data[:,K+i] * ((new_data[:,P+i]/365.25)**(1./3.)) * (1./28.4) /1000
        pl_mass = np.percentile(mass,50)
        pl_massup = np.percentile(mass,84)-np.percentile(mass,50)
        pl_masslow = np.percentile(mass,50)-np.percentile(mass,16)
        print('Mass (M Jupiter)= {0} +{1} -{2}'.format(pl_mass,pl_massup,pl_masslow))
        print('Mass (M Earth)= {0} +{1} -{2}'.format(pl_mass*317.8,pl_massup*317.8,pl_masslow*317.8))
        print()














