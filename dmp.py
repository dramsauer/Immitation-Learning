import numpy

def generate_trajectory(S,G,T,w,s0):
    
    #generate kernel centers
    c=numpy.linspace(0,1,len(w))
    #generate goal function
    goal=numpy.linspace(S,G,T)
    
    y=S  #initial position
    Y=[] #trajectory vector
    Y.append(y)
    z=0  #intial velocity
    dz=0 #initial accelleration
    
    #contstants of dmp framework
    alpha_z=0.75
    beta_z=alpha_z/4
           
    for t in range(1, T):
        #generate kernels psi
        b=float(t)/(T-1)-c
        psi=numpy.exp(-b*b/(2*s0*s0))
        
        #generate f
        f=sum(psi*w)/sum(psi)

        #compute accelleration
        dz=alpha_z*(beta_z*(goal[t]-y)-z)+f
        #compute velocity
        z=z+dz
        #compute position
        y=y+z
        #add point to the trajectory vector
        Y.append(y)
        
    return Y
