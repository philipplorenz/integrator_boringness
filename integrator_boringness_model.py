import numpy as np
import scipy.integrate
import sys
import matplotlib.pyplot as plt
import pickle

#the model
def system(t, y, a, c, r):
	
	L = [y[i*2] for i in range(len(y)/2)]
	Y = [y[i*2+1] for i in range(len(y)/2)]
		
	dydt = []
	l_sum = sum(L)
	for i in range(len(y)/2):
		
		dydt.append(1.0*r*L[i]*(1.0 - (r*Y[i] + c*(l_sum-L[i]))))
		dydt.append(L[i] - a*Y[i])

	return dydt

#parameters from command line
a = float(sys.argv[1]) #0.005
c = float(sys.argv[2]) #2.4
r = float(sys.argv[3]) #12.0
size = int(sys.argv[4]) #300
#length and granularity of the solution
t_end = 10000.
t_start = 0.
t_step = 1.0
t_interval = np.arange(t_start, t_end+t_step, t_step)

#random intital conditions
y0 = []
for i in range(size):
	
	y0.append(np.random.random()*1)
	y0.append(0.0)

#setting up the integrator
ode = scipy.integrate.ode(system)
ode.set_integrator('dopri5')
ode.set_initial_value(y0, t_start).set_f_params(a, c, r)

#running the solver
ts = []
sol = []
while ode.successful() and ode.t < t_end:

	ode.integrate(ode.t + t_step)
	ts.append(ode.t)
	sol.append(ode.y)

t = np.vstack(ts)

#processing the results, binning accoring to stepsize and ploting
trajectories = []
y_trajectories = []

for i in range(len(y0)/2):
	
	traj1 = [s[i*2] for s in sol]
	y_traj1 = [s[i*2+1] for s in sol]
	#plot the trajectories for checking
	trajectories.append(traj1)
	plt.plot(traj1, lw=2)
	y_trajectories.append(y_traj1)
	#plt.plot(y_traj1, lw=2)

plt.xlabel('$t$', fontsize=20)
plt.ylabel('$L_i(t)$', fontsize=20)
plt.show()
