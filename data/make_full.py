import numpy as np 
import glob


files = glob.glob('RV_K2-19*.txt')
assert len(files) == 6


data = np.empty((0,4))

for i, f in enumerate(files[:4]):
	print f
	t,y,e = np.loadtxt(f, unpack=True, skiprows=2)
	ind = (i+1)*np.ones_like(t)
	data = np.append(data, np.array([t,y,e,ind]).T, axis=0)


#### sort by time!
data = data[data[:,0].argsort()]


np.savetxt('K2_19_full.txt', data,
	       header='time\tvrad\tsvrad\n----\t----\t-----',
	       fmt=['%12.6f', '%7.5f', '%7.5f', '%d'])


# 57395.856817	7.33907	0.00814