#importing relevant packages
import numpy as np
from types import SimpleNamespace
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')




def h(x, parlist):
    """ The function defines and sets up the model as a matrix
    Args:
         x: a vector containing endogenous variables
         parlist: a list with parameters
         
        Returns:
        The deviation from equality

    """
    #maps variable names to vector elements
    par=parlist
    eq=np.zeros(18)
    L1=x[0]
    L2=x[1]
    p1=x[2]
    p2=x[3]
    Y1=x[4]
    Y2=x[5]
    M1=x[6]
    M2=x[7]
    pm1=x[8]
    pm2=x[9]
    x11=x[10]
    x12=x[11]
    x21=x[12]
    x22=x[13]
    C1=x[14]
    C2=x[15]
    pc=x[16]
    YD=x[17]
    
    #labour demand function
    eq[0] = par.theta1*L1-par.muYL1*(par.w/(par.theta1*p1))**(-par.EY)*Y1
    eq[1] = par.theta2*L2-par.muYL2*(par.w/(par.theta2*p2))**(-par.EY)*Y2
    
    #materials demand function
    eq[2] = par.theta1m*M1-par.muYM1*(pm1/(par.theta1m*p1))**(-par.EY)*Y1
    eq[3] = par.theta2m*M2-par.muYM2*(pm2/(par.theta2m*p2))**(-par.EY)*Y2
    
    # 0-profit assumption function
    eq[4] = pm1*M1+par.w*L1-p1*Y1
    eq[5] = pm2*M2+par.w*L2-p2*Y2
    
    #input materials demand function
    eq[6] = x11-par.mux11*(p1/pm1)**(-par.EM)*par.theta1m*M1
    eq[7] = x12-par.mux12*(p1/pm2)**(-par.EM)*par.theta2m*M2
    
    #input materials demand function
    eq[8] = x21-par.mux21*(p2/pm1)**(-par.EM)*par.theta1m*M1
    eq[9] = x22-par.mux22*(p2/pm2)**(-par.EM)*par.theta2m*M2
    
    #equilibrium function for input materials
    eq[10] = pm1*M1-p1*x11-p2*x21
    eq[11] = pm2*M2-p1*x12-p2*x22

    #consumer demand function
    eq[12]= C1-par.gamma1*(p1/pc)**(-par.EC)*YD/pc
    eq[13]= C2-par.gamma2*(p2/pc)**(-par.EC)*YD/pc
    
    #equilibrium function for goods
    eq[14] = YD-p1*C1-p2*C2
    
    #income
    eq[15] = YD-par.w*par.N
    
    #goods equilibrium function
    eq[16] = Y1 - x11-x12 - C1
    eq[17] = Y2 - x21-x22 - C2

    return eq




def calibrate(x, parlist):
    """ The function isolates each scale parameter
    Args:
         x: a vector containing endogenous variables
         parlist: a list with parameters
         
    Returns:
       parameter values that fit the mock data 

    """
    par=parlist
    print(f'Equation values before calibration: {h(x, par)}')
        #Calibrate labour input demand - 
    par.muYL1=par.theta1*x[0]/((par.w/(par.theta1*x[2]))**(-par.EY)*x[4])
    par.muYL2=par.theta2*x[1]/((par.w/(par.theta2*x[3]))**(-par.EY)*x[5])
    par.muYM1 =x[6]/((x[8]/x[2])**(-par.EY)*x[4])
    par.muYM2 = x[7]/((x[9]/x[3])**(-par.EY)*x[5])
    par.mux11 =  x[10]/((x[2]/x[8])**(-par.EM)*x[6])
    par.mux12 =  x[11]/((x[2]/x[9])**(-par.EM)*x[7])
    par.mux21 =  x[12]/((x[3]/x[8])**(-par.EM)*x[6])
    par.mux22 =  x[13]/((x[3]/x[9])**(-par.EM)*x[7])
    par.gamma1 = x[14]/((x[2]/x[16])**(-par.EC)*x[17]/x[16])
    par.gamma2 = x[15]/((x[3]/x[16])**(-par.EC)*x[17]/x[16])
    print(f'Equation values after calibration: {h(x, par)}')
    return par

    
    
def solve_model(x0, EC, EM, EY, theta1, theta2, parlist, theta1m, theta2m):
    """ The function solves the model by solving the model using a root finder
    Args:
         x0: initial vector if the guess is poor the model won't be solved
         EC: parameters included in order to be changed
         EM: parameters included in order to be changed
         EY: parameters included in order to be changed
         theta1: parameters included in order to be changed
         theta2: parameters included in order to be changed
         theta1m: parameters included in order to be changed
         theta2m: parameters included in order to be changed
         parlist: a list with parameters
         
    Returns:
       the solution vector x

    """
    par=parlist
    par.EC=EC
    par.EM=EM
    par.EY=EY
    par.theta1=theta1
    par.theta2=theta2
    par.theta1m=theta1m
    par.theta2m=theta2m                                
    result = optimize.root(h,x0, args=(par))
    #Check whether solution is found
    if result.success==False:
        raise Exception("Solution not found")

    return result.x 
    
def create_timeseries(EC1, EM1, EY1, x01, parlist):
    """ This function serves as an interactive visualization tool showing how labour changes 
        in sector 1 & 2 if we change the elasticities of substitution (EC, EM, EY)
        
   Args:
         x0: initial vector if the guess is poor the model won't be solved
         EC1: parameters included in order to be changed
         EM1: parameters included in order to be changed
         EY1: parameters included in order to be changed
         parlist: a list with parameters        
    """ 
    x0=x01
    a=[]                                              #empty list to append the loop results in
    for i in range(1,300):
        Grow_theta1=(1+0.03)**(i-1)
        Grow_theta2= (1+0.01)**(i-1)
        Grow_theta1m=1
        Grow_theta2m=1
        result_loop = solve_model(x0, EC1, EM1, EY1, Grow_theta1, Grow_theta2, parlist, Grow_theta1m, Grow_theta2m)
        x0=result_loop
        a.append(result_loop[0:2])                    #setting the variables of interest (L1, L1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_yticks(range(0,1000, 100))                 #fixing where the ticks are on y axis

# plotting the data
    ax.plot(a)

# naming the axes, setting legend and title
    plt.xlabel('Period')
    plt.ylabel('Labour')
    ax.legend(['sector 1', 'sector 2'])

    plt.title('Evolution of labour')
    plt.ylim((0,1050)) #setting the limits for the graph

    
    
def graph_consumption(EC1, EM1, EY1, x01, parlist):
    """ This function serves as an interactive visualization tool showing how consumption changes 
        in sector 1 & 2 if we change the elasticities of substitution (EC, EM, EY)
        
   Args:
         x0: initial vector if the guess is poor the model won't be solved
         EC: parameters included in order to be changed
         EM: parameters included in order to be changed
         EY: parameters included in order to be changed
         parlist: a list with parameters        
    """ 
    x0=x01
    b=[]                                              #empty list to append the loop results into
    for i in range(1,300):
        Grow_theta1=(1+0.03)**(i-1)
        Grow_theta2= (1+0.01)**(i-1)
        Grow_theta1m=1
        Grow_theta2m=1
        result_loop2 = solve_model(x0, EC1, EM1, EY1, Grow_theta1, Grow_theta2, parlist, Grow_theta1m, Grow_theta2m)
        x0=result_loop2
        b.append(result_loop2[14:16])                #setting the variables of interest (C1, C2) 
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

# plotting the data
    ax.plot(b)

# naming the axes, setting legend and title
    plt.xlabel('Period')
    plt.ylabel('Consumtion')
    ax.legend(['sector 1 good', 'sector 2 good'])

    plt.title('Evolution of consumption')
    
    


def create_timeseries_L_M1(EC1, EM1, EY1, x01, parlist):
    """ This function serves as an interactive visualization tool showing how elasticity of substitution between materials and labour changes 
        in sector 1 if we change the elasticities of substitution (EC, EM, EY)
        
    
   Args:
         x0: initial vector if the guess is poor the model won't be solved
         EC: parameters included in order to be changed
         EM: parameters included in order to be changed
         EY: parameters included in order to be changed
         parlist: a list with parameters    
    """ 
    x0=x01
    a=[]                                            #empty list to append the loop results into
    for i in range(1,300):
        Grow_theta1=(1+0.03)**(i-1)
        Grow_theta2= (1+0.01)**(i-1)
        Grow_theta1m=1
        Grow_theta2m=1
        result_loop = solve_model(x0, EC1, EM1, EY1, Grow_theta1, Grow_theta2, parlist, Grow_theta1m, Grow_theta2m)
        x0=result_loop
        a.append([result_loop[0], result_loop[6]/((1+0.03)**(i-1))])  #setting the variables of interest (L1, M1 Growth corrected)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

# plotting the data
    ax.plot(a)

# naming the axes, setting legend and title
    plt.xlabel('Period')
    plt.ylabel('Labour, Materials(Growth corrected)')
    ax.legend(['Labour', 'Materials'])
#     plt.ylim((0,1000))
    
    plt.title('Materials(Growth corrected) and labour - sector 1')
    
    
    
    
    
def graph_x(EC1, EM1, EY1, x01, parlist):
    """ This function serves as an interactive visualization tool showing how material inputs change 
        in sector 1 and 2, in two separate graphs, if we change the elasticities of substitution (EC, EM, EY)
   Args:
         x0: initial vector if the guess is poor the model won't be solved
         EC: parameters included in order to be changed
         EM: parameters included in order to be changed
         EY: parameters included in order to be changed
         parlist: a list with parameters        
    """ 
    x0=x01
    c=[]
    d=[]
    for i in range(1,300):
        Grow_theta1=(1+0.03)**(i-1)
        Grow_theta2= (1+0.01)**(i-1)
        Grow_theta1m=1
        Grow_theta2m=1
        result_loop3 = solve_model(x0, EC1, EM1, EY1, Grow_theta1, Grow_theta2, parlist, Grow_theta1m, Grow_theta2m)
        x0=result_loop3
        c.append(result_loop3[10:14:2])                    #setting the variables of interest (x11, x21) for sector 1
        d.append(result_loop3[11:15:2])                    #setting the variables of interest (x11, x21) for sector 2
    fig, axs = plt.subplots(2)

   # plotting the data
    axs[0].plot(c)
    axs[0].set_title('sector 1 material inputs')
    axs[0].legend(['sector 1 good', 'sector 2 good'])
    axs[1].plot(d)
    axs[1].set_title('sector 2 material inputs')
    axs[1].legend(['sector 1 good', 'sector 2 good'])

   # naming the axes
    plt.xlabel('Period')
    fig.tight_layout()

                                      

                                      
def create_timeseries_L_M1_g(EC1, EM1, EY1, x01, parlist):
    """ This function serves as an interactive visualization tool showing how elasticity of substitution between materials and labour changes 
        in sector 1 if we change the elasticities of substitution (EC, EM, EY)
        
    
   Args:
         x0: initial vector if the guess is poor the model won't be solved
         EC: parameters included in order to be changed
         EM: parameters included in order to be changed
         EY: parameters included in order to be changed
         parlist: a list with parameters    
    """ 
    x0=x01
    a=[]                                            #empty list to append the loop results into
    for i in range(1,200):
        Grow_theta1=(1+0.03)**(i-1)
        Grow_theta2= (1+0.1)**(i-1)
        Grow_theta1m=(1+0.03)**(i-1)
        Grow_theta2m= (1+0.01)**(i-1)
        result_loop = solve_model(x0, EC1, EM1, EY1, Grow_theta1, Grow_theta2, parlist, Grow_theta1m, Grow_theta2m)
        x0=result_loop
        a.append([result_loop[0], result_loop[6]])  #setting the variables of interest (L1, M1 Growth corrected)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

# plotting the data
    ax.plot(a)

# naming the axes, setting legend and title
    plt.xlabel('Period')
    plt.ylabel('Labour, Materials')
    ax.legend(['Labour', 'Materials'])
#     plt.ylim((0,1000))
    
    plt.title('Materials and labour - sector 1')

def create_timeseries_L_M2_g(EC1, EM1, EY1, x01, parlist):
    """ This function serves as an interactive visualization tool showing how elasticity of substitution between materials and labour changes 
        in sector 1 if we change the elasticities of substitution (EC, EM, EY)
        
    
   Args:
         x0: initial vector if the guess is poor the model won't be solved
         EC: parameters included in order to be changed
         EM: parameters included in order to be changed
         EY: parameters included in order to be changed
         parlist: a list with parameters    
    """ 
    x0=x01
    a=[]                                            #empty list to append the loop results into
    for i in range(1,200):
        Grow_theta1=(1+0.03)**(i-1)
        Grow_theta2= (1+0.1)**(i-1)
        Grow_theta1m=(1+0.03)**(i-1)
        Grow_theta2m= (1+0.01)**(i-1)
        result_loop = solve_model(x0, EC1, EM1, EY1, Grow_theta1, Grow_theta2, parlist, Grow_theta1m, Grow_theta2m)
        x0=result_loop
        a.append([result_loop[1], result_loop[7]])  #setting the variables of interest (L1, M1 Growth corrected)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

# plotting the data
    ax.plot(a)

# naming the axes, setting legend and title
    plt.xlabel('Period')
    plt.ylabel('Labour, Materials')
    ax.legend(['Labour', 'Materials'])
#     plt.ylim((0,1000))
    
    plt.title('Materials and labour - sector 1')

def create_timeseries_Y1_Y2(EC1, EM1, EY1, x01, parlist):
    """ This function serves as an interactive visualization tool showing how elasticity of substitution between materials and labour changes 
        in sector 1 if we change the elasticities of substitution (EC, EM, EY)
        
    
   Args:
         x0: initial vector if the guess is poor the model won't be solved
         EC: parameters included in order to be changed
         EM: parameters included in order to be changed
         EY: parameters included in order to be changed
         parlist: a list with parameters    
    """ 
    x0=x01
    a=[]                                            #empty list to append the loop results into
    for i in range(1,100):
        Grow_theta1=(1+0.03)**(i-1)
        Grow_theta2= (1+0.1)**(i-1)
        Grow_theta1m=1
        Grow_theta2m= 1
        result_loop = solve_model(x0, EC1, EM1, EY1, Grow_theta1, Grow_theta2, parlist, Grow_theta1m, Grow_theta2m)
        x0=result_loop
        a.append([result_loop[4], result_loop[5]])  #setting the variables of interest (L1, M1 Growth corrected)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

# plotting the data
    ax.plot(a)

# naming the axes, setting legend and title
    plt.xlabel('Period')
    plt.ylabel('Output')
    ax.legend(['Sector 1', 'Sector 2'])
#     plt.ylim((0,1000))
    
    plt.title('Output')