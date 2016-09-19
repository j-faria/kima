import sys
import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import linregress

if __name__ == '__main__':
    if __package__ is None:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.data_handler import get_system
    else:
        from scripts.data_handler import get_system


if __name__ == '__main__':
    try:
        i = int(sys.argv[1])
        system = get_system(number=i, ms=False)
        # make_correlation_plot(system)
    except IndexError, e:
        print __file__, 'system_number'
        sys.exit(1)

# system = get_system(number=1, ms=False)

t = system.time
res = linregress(t-t[0], system.vrad)
m, b = res.slope, res.intercept
print 'slope:', m, 'intercept (at t[0]):', b


max_rv = system.vrad.max()
time_max_rv = system.time[system.vrad.argmax()]

min_rv = system.vrad.min()
time_min_rv = system.time[system.vrad.argmin()]

max_slope = (max_rv - min_rv) / (time_max_rv - time_min_rv)
print 'max_slope', max_slope

min_slope = -max_slope # (min_rv - max_rv) / (time_max_rv - time_min_rv)
print 'min_slope', min_slope



system.do_plot_obs()
plt.plot(system.time, m*(t-t[0]) + b, '-r', lw=3)
# for _ in range(500):
# 	# mm = np.random.normal(loc=0, scale=m)
# 	mm = np.random.uniform(low=min_slope, high=max_slope)
# 	plt.plot(system.time, mm*(t-t[0]) + b, '-k', lw=1, alpha=0.3)
plt.show()


old_file = system.provenance.keys()[0]
old_path = os.path.dirname(old_file)
new_file = os.path.basename(old_file)[:-3] + 'noslope.rdb'

print 'Remove this linear trend from the data',
print 'and save it as %s ?' % new_file, 
print '(y/n)', 
yn = raw_input()
if yn == 'y':
	line = m*(t-t[0]) + b
	system.vrad -= line
	system.do_plot_obs()
	plt.show()
	X = [system.time, system.vrad, system.error]
	header = 'jdb\tvrad\tsvrad\n---\t----\t-----'
	np.savetxt(os.path.join(old_path, new_file), zip(*X), 
		       fmt=['%12.6f', '%8.5f', '%7.5f'], delimiter='\t', header=header, comments='')

else:
	print 'Doing nothing. Bye!'