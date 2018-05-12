
import dmp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intrp

trj = np.loadtxt('../target_trajectory.dat')
T=len(trj)  #movement duration (in steps)

Sx=trj[0,0]   #start X
Sy=trj[0,1]   #start Y
Gx=trj[T-1,0] #goal X
Gy=trj[T-1,1] #goal Y

nk=T       #number of kernels
s0=0.0001    #kernel width    
ne=100      #number of learning epochs
lr=0.1      #learning rate

#intiliaze kernels with zeros
wx=np.zeros(nk)
wy=np.zeros(nk)    

#generate indices for weights
k=np.round(np.linspace(0,T-1,nk)).astype(int)

#learn weights using delta rule    
for e in range(0, ne):
    #generate X and Y trajectories based on current weights
    X=np.array(dmp.generate_trajectory(Sx,Gx,T,wx,s0))
    Y=np.array(dmp.generate_trajectory(Sy,Gy,T,wy,s0))
    
    #update weights
    wx=wx+lr*(trj[k,0]-X[k])
    wy=wy+lr*(trj[k,1]-Y[k])


#generate learnt trajectory with different end-point
X=dmp.generate_trajectory(Sx,Gx+10,T,wx,s0)
Y=dmp.generate_trajectory(Sy,Gy+10,T,wy,s0)


#normalize to zero start point
X=X-X[0]
Y=Y-Y[0]
trj=trj-trj[0,:]

#scale trajectory
a=0.0005
X=a*X
Y=a*Y
trj=a*trj

#signal upsampling via interpolation
t=np.arange(0,T)
t_new=np.arange(0,T-1,0.05)
fx = intrp.interp1d(t, X, 'cubic')
fy = intrp.interp1d(t, Y, 'cubic')
X = fx(t_new)
Y = fy(t_new)

#set all Z coordinate values to zeros
Z=np.zeros(len(X))

#save trajectory
data=np.array([X,Y,Z])
np.savetxt('trajectory.dat', data.T, fmt="%12.6f %12.6f %12.6f")

#plot trajectories
plt.plot(trj[:,0],trj[:,1],'b',label='Target trajectory')
plt.plot(X,Y,'r',label='Learnt trajectory')
plt.xlabel('Trajectory X')
plt.ylabel('Trajectory Y')
ax = plt.subplot(111)
ax.legend()
plt.show()

#compute squared velocity
V=np.sqrt( np.array(pow(np.array(np.diff(X)),2) + pow(np.array(np.diff(Y)),2)) )
plt.plot(V)
plt.show()



