import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def Euler(dt, f, t, y, args):
    return y + f(t,y,*args) * dt 

def EulerCromer(dt, f, t, y, args): #go full step forward and sample that for interval
    y_end = y + f(t,y,*args) * dt
    return y + f(t+dt,y_end,*args) * dt


def EulerRichardson(dt, f, t, y, args): #Midpoint sampling
    y_mid = y + f(t, y, *args) * (dt / 2) #Gives us the velocity at the midpoint
    return y + f(t + dt/2, y_mid, *args) * dt #We calculate overall difference using midpoint velocity. y + velocity at midpoint * dt

def OtherEuler(dt, f, t, y, args):
    y_mid = y + f(t, y, *args) * (dt / 2) #Gives us the velocity at the midpoint
    
    return y + EulerRichardson(t + dt/2, y_mid, *args) * dt

def RungeKutta4(dt,f,t,y,args):

    h = dt
    k1 = Euler(dt,f,t,y)
    k2 = EulerRichardson(dt,f,t,y)
    k3 = OtherEuler(dt,f,t,y)
    


def solve_ode(f, tspan, y0, method=EulerRichardson, *args, **options):
    """
    Given a function f that returns derivatives,
    dy / dt = f(t, y, *args)
    and an initial state:
    y(tspan[0]) = y0
    
    This function will return the set of intermediate states of y
    from t0 (tspan[0]) to tf (tspan[1])
    
    first_step: dt
    y_ground: y position to end at
    """

    dt = options.get('first_step', 0.1)
    y_ground = options.get('y_ground', None)
    
    numsteps = int((tspan[1] - tspan[0]) / dt)

    y = []
    y.append(y0)
    
    t = []
    t.append(tspan[0])
    
    has_risen = False
    
    for i in range(1, numsteps):
        t.append(tspan[0] + i * dt)
        y.append(method(dt, f, t[i-1], y[i-1], args))
        
        #This is specific to ballistics and cuts off the simulation when the bullet hits the y target/ground
        if y_ground is not None:
            if y[-1][1] > y_ground:
                has_risen = True
            if has_risen and y[-1][1] <= y_ground:
                break
        
    return np.array(y), np.array(t)

def gravitational(rij,i,j,p):
    """
    Using the dictionary of parameters p return the gravitational force between particles i
    and j. We will assume that the vector between them has already been computed as rij.

    Returns a vector force, having d components where d is the dimension of the problem.
    
    """
    return rij * -p['G'] * p['m'][i] * p['m'][j] / (norm(rij)**3)
    

def n_body(t,y,p):
    """
    Develop the code that accepts a state vector y for N particles in d dimensions.
    For example, in 3 dimensions:
    y = [x1, y1, z1, x2, y2, z2, ... vx1, vy1, vz1, vx2, vy2, vz2,...]
    Normally, this would be up to you, but for this assignment it
    is better you follow my lead, because I have many interesting 
    initial conditions (y0) for you to explore.

    The function should then compute dydt for each of the input values.
    It should do so using a force function that computes the magnitude of force.
    That function should be a function handle in parameter dictionary p.

    The force must be compute between every pair of particles i,j
    This requires nested loops. 
    Record the forces in a matrix. 
    The sum of each row of the matrix corresponds to the total force on a particle.
    The dictionary p will contain any and all additional parameters needed in the calculation.
    """
    masses = p['m']
    N = len(masses)
    d = p['dimension']
    FMatrix = np.zeros((N,N,d))
    midpoint = len(y)//2
    
    dydt = np.zeros(len(y))
    dydt[0:midpoint] = y[midpoint:]     
    
    pos = np.array(y[:N*d]).reshape(N,d)
    #print(pos)
    #print(pos.shape)
    
    for i in range(N):
        #Identify ith position vector
        vi = pos[i]
        
        for j in range(i+1,N): 
            #jth position vector
            vj = pos[j]
            rij = vi-vj
        
            fij = p['force'](rij,i,j,p)
            
            FMatrix[i,j] = fij
            FMatrix[j,i] = -fij

    #Sum forces on ith vector and map to dydt    
    #Use sum to do this in one step not loop
    for i in range(N):
        '''i_dydt = np.sum(FMatrix[i],axis=0)
        print(i_dydt)
        print(i_dydt.shape)
        print("dydt")
        print(dydt)
        print(dydt.shape)'''
        
        #set velocity half of to force taken from FMatrix, divide my masses to get force because F=ma -> a = F/m
        dydt[midpoint + i*d:midpoint + (i+1)*d] = np.sum(FMatrix[i],axis=0)/masses[i]
    
    '''print(FMatrix)
    print(FMatrix.shape)
    print(f"dydt:{dydt}")'''
    return dydt
