import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import pandas as pd
import json
import pickle
import scipy.integrate

def f(t, y, params):
    L = y[:len(y)/2]
    Y = y[len(y)/2:]
    a, c, r, K = params  # unpack parameters
    
    l_sum = sum(L)
    dL = []
    dY = []
    for i in range(len(L)):
	dL.append(r*L[i]*(1.0 - ((r/K)*Y[i] + c*(l_sum-L[i]))))
        dY.append(L[i] - a*Y[i])
	
    derivs = dL + dY
    return derivs

# Parameters
a = float(sys.argv[1]) #0.005
c = float(sys.argv[2]) #2.4
r = float(sys.argv[3]) #11.0
K = float(sys.argv[4]) #1.0
size = int(sys.argv[5]) #300

rel_changes = []
log_changes = []

for trial in range(1):

	L = []
	Y = []
	# Initial values
	for i in range(size):
		L.append(np.random.random())	# initial values
		Y.append(0.0)     		# initial memories

	# Bundle parameters for ODE solver
	params = [a, c, r, K]

	# Bundle initial conditions for ODE solver
	y0 = L + Y

	# Make time array for solution
	tStop = 2000
	tInc = 0.05	
	t = np.arange(0., tStop, tInc)

	# Call the ODE solver
	#psoln = odeint(f, y0, t, args=(params,))

	ode = scipy.integrate.ode(f)
	ode.set_integrator('dopri5')
	ode.set_initial_value(y0, 0.0).set_f_params(params)

	#running the solver
	ts = []
	psoln = []
	while ode.successful() and ode.t < tStop:
		ode.integrate(ode.t + tInc)
		ts.append(ode.t)
		psoln.append(ode.y)

	t = np.vstack(ts)
	psoln = np.array(psoln)		
	
	traj_list = []
	for i in range(size):
		traj_list.append(list(psoln[:,i]))
		plt.plot(psoln[:,i][10000:])	

	for step in range(10000, len(psoln[:,i])):
		
		if step%20 ==0:		

			for traj in traj_list[:]:
			
				relative_change = (traj[step] - traj[step-20])/traj[step-20]
				if relative_change > 0.00:
		    			rel_changes.append(relative_change)
				if traj[step-20] > 0:
					log_changes.append(np.log(traj[step]/traj[step-20]))

#with open("./trajectory_{0}.json".format(r), "wb") as fp:   #Pickling
#	json.dump(list(traj_list), fp)

#with open("./relative_change_data_{0}.json".format(r), "wb") as fp:   #Pickling
#	json.dump(rel_changes[:], fp)

#with open("./logarithmic_change_data_{0}.json".format(r), "wb") as fp:   #Pickling
#	json.dump(log_changes[:], fp)

plt.xlim(28000, 30000)
plt.show()

data = pd.Series(rel_changes)
# Plot for comparison
ax = data.plot(kind='hist', normed=True, alpha=0.3, label='Simulation', color='red', loglog=True, bins=np.logspace(np.log10(min(rel_changes)),np.log10(max(rel_changes)), 50))

with open("./example_data_twitter_2016.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)
x_new2=[]
for relhype in b:
	if relhype > 0.00:
            x_new2.append(abs(relhype))

data2 = pd.Series(x_new2)
ax = data2.plot(kind='hist', normed=True, alpha=0.3, label='Data', color='black', loglog=True, bins=np.logspace(np.log10(min(x_new2)),np.log10(max(x_new2)), 50))
plt.legend()
plt.show()
