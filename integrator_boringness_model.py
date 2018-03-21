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
		
		dydt.append(r*L[i]*(1.0 - (Y[i]  + c*(l_sum-L[i]))))
		dydt.append(L[i] - a*Y[i])

	return dydt

#parameters from command line
a = float(sys.argv[1])
c = float(sys.argv[2])
r = float(sys.argv[3])
size = int(sys.argv[4])
#length and granularity of the solution
t_end = 4000.
t_start = 0.
t_step = 1.0
t_interval = np.arange(t_start, t_end+t_step, t_step)
#random intital conditions
y0 = []
for i in range(size):

	y0.append(0.01*np.random.random())
	y0.append(0.0)
#setting up the integrator
ode = scipy.integrate.ode(system)
ode.set_integrator('dopri5', nsteps=1000)
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
for i in range(len(y0)/2):
	
	traj1 = [s[i*2] for s in sol]
	plt.plot(traj1, lw=3)
	traj = []
	for t in range(len(traj1)):
		if t%(1.0/t_step) == 0:
			traj.append(sum(traj1[t:t+int(1.0/t_step)]))
	trajectories.append(traj)

plt.xlabel('$t$', fontsize=20)
plt.ylabel('$L_i(t)$', fontsize=20)
plt.show()

#collect relative changes for saving
final_list = [[] for t in range(len(trajectories[0]))]
final_list_reverse = [[] for t in range(len(trajectories[0]))]
for traj in trajectories:

	if max(traj) > 0.0:	
		for j in range(len(traj)-1):

			if j > 1:

				if j != 0 and traj[j-1] > 0.0 and traj[j+1] > 0.0:
				
					change = (traj[j] - traj[j-1])/float(traj[j-1])
					final_list[j/1].append([i, traj[j], change])
					change_reverse = (traj[j] - traj[j+1])/float(traj[j+1])
					final_list_reverse[j/1].append([i, traj[j], change_reverse])

#save data using pickle in various files
with open("./trajectory_simulated_{0}.txt".format(r), "wb") as fp:   #Pickling
	pickle.dump(trajectories[:], fp)
with open("./plotdata_simulated_{0}.txt".format(r), "wb") as fp:   #Pickling
	pickle.dump(final_list[:], fp)
with open("./plotdata_simulated_reverse_{0}.txt".format(r), "wb") as fp:   #Pickling
	pickle.dump(final_list_reverse[:], fp)



