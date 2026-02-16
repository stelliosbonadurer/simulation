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
        
        #This is specific to the ballistics notebook and cuts off the simulation when the bullet hits the y target/ground
        if y_ground is not None:
            if y[-1][1] > y_ground:
                has_risen = True
            if has_risen and y[-1][1] <= y_ground:
                break
        
    return np.array(y), np.array(t)

