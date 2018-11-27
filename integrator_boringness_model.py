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
    a, c, r = params  # unpack parameters
    
    l_sum = sum(L)
    dL = []
    dY = []
    for i in range(len(L)):
	dL.append(r*L[i]*(1.0 - (r*Y[i] + c*(l_sum-L[i]))))
        dY.append(L[i] - a*Y[i])
	
    derivs = dL + dY
    return derivs

# Parameters
a = float(sys.argv[1]) #0.005
c = float(sys.argv[2]) #2.4
r = float(sys.argv[3]) #11.0
size = int(sys.argv[4]) #100

x_new = []

for trial in range(1):

	L = []
	Y = []
	# Initial values
	for i in range(size):
		L.append(np.random.random())	# initial values
		Y.append(0.0)     		# initial memories

	# Bundle parameters for ODE solver
	params = [a, c, r]

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
		traj_list.append(psoln[:,i])
		plt.plot(psoln[:,i][:])	

	for step in range(20, len(psoln[:,i])):
		
		traj_list.sort(key=lambda x: x[step], reverse=True)
		if step%20 ==0:		

			for traj in traj_list[:20]:
			
				#relative_change = traj[step]
				relative_change = (traj[step-20] - traj[step])/traj[step]
				if relative_change > 0.00:
		    			x_new.append(relative_change)

#with open("./relative_change_data_{0}.json".format(r), "wb") as fp:   #Pickling
#	json.dump(x_new[:], fp)

plt.show()

data = pd.Series(x_new)
# Plot for comparison
ax = data.plot(kind='hist', normed=True, alpha=0.3, label='Simulation', color='red', loglog=True, bins=np.logspace(np.log10(min(x_new)),np.log10(max(x_new)), 50))

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
