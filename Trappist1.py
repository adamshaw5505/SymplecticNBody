from nbody_sym import World, Object
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    objects = []

    ''' Trappist 1 '''
    
    m0 = 5.972e24 # Mass of earth
    au = 1.495979e11 # 1 Astranomical Unit
    T_day = 3600*24 # 1 Day in seconds
    T = 19*T_day # 19 Earth Days
    G = 6.67e-11 # Actual value of G
    TAmount = 64000
    #Initial conditions for Trappist-1 (calculated velocities by hand) 
    objects.append(Object([0,0,0],[0,0,0],29641.8285332*m0)) # Star (Trappist-1a)
    objects.append(Object([0.01150*au, 0, 0], [0, 82806.9198985, 0], 1.017*m0)) # Trappist-1b
    objects.append(Object([0.01576*au, 0, 0], [0, 70680.6448923, 0], 1.156*m0)) # trappist-1c
    objects.append(Object([0.02219*au, 0, 0], [0, 59510.0860787, 0], 0.297*m0)) # Trappist-1d
    objects.append(Object([0.02916*au, 0, 0], [0, 51928.9722344, 0], 0.772*m0)) # Trappist-1e
    objects.append(Object([0.03836*au, 0, 0], [0, 45259.6414294, 0], 0.934*m0)) # Trappist-1f
    objects.append(Object([0.04670*au, 0, 0], [0, 41055.9928269, 0], 1.148*m0)) # Trappist-1g
    objects.append(Object([0.06170*au, 0, 0], [0, 35706.9010221, 0], 0.331*m0)) # Trappist-1h

    #Setup the world and add the objects
    world = World()
    world.SetG(G)
    world.SetTAmount(TAmount)
    world.Add(objects)
    
    '''Plotting Trajectories'''
    fig = plt.figure(2)
    ax1 = fig.add_subplot(111)
    world.CalcTrajectories(T,World.fourthsymp)
    world.SetAxisLimits(ax1)
    [o.Draw(T,False,ax1) for o in world.objects]
    plt.show()

    
    '''Getting Period Times'''
    fig = plt.figure(2)
    ax1 = fig.add_subplot(111)
    world.CalcTrajectories(T,World.fourthsymp)
    world.SetAxisLimits(ax1)
    print("Periods: ", np.asarray(world.Periods)/T_day)
    plt.show()

    '''Comparing Energies Of Different Methods'''
    energy_methods = [("Symplectic-4th", World.fourthsymp),("Adaptive RK45", World.rk45)]
    energy_list = []
    legend_list = []
    period_list = []
    print("Δt: {0:g}".format(T/TAmount))
    for l,m in energy_methods:
        print("Calculating: {0:}".format(l))
        world.CalcTrajectories(T,m)
        e = world.Energy()
        print("ΔE/E0: ", np.abs(np.max(e)-np.min(e)))
        print("Periods: ", world.Periods)
        legend_list.append(l)
        energy_list.append(world.Energy())
    fig = plt.figure(3)
    ax = fig.add_subplot(111)

    t_range = world.Times
    for e_range in energy_list:
        ax.plot(t_range,e_range)
    ax.legend(legend_list)
    plt.ylabel("ΔE/E0")
    plt.xlabel("t")
    plt.show()

    '''Energies for longer times (T = 190 days)'''
    world.SetTAmount(256000)
    world.CalcTrajectories(190*T_day,World.fourthsymp)
    symp_e = world.Energy()
    world.CalcTrajectories(190*T_day,World.rk45)
    rk45_e = world.Energy()

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    t_range = world.Times
    ax.plot(t_range,symp_e)
    ax.plot(t_range,rk45_e)
    ax.legend(["Symplectic 4-th","Adaptive RK45"])
    plt.show()

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    t_range = world.Times
    ax.plot(t_range,symp_e)
    plt.show()

    '''Energies for longer times (T = 19000 days)'''
    world.SetTAmount(256000)
    world.CalcTrajectories(19000*T_day,World.fourthsymp)
    symp_e = world.Energy()
    print("Energy Range: ", np.abs(np.max(symp_e)-np.min(symp_e)))
    world.CalcTrajectories(19000*T_day,World.rk45)
    rk45_e = world.Energy()
    print("Energy Range: ", np.abs(np.max(rk45_e)-np.min(rk45_e)))
    

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    t_range = world.Times
    ax.plot(t_range,symp_e)
    ax.plot(t_range,rk45_e)
    ax.legend(["Symplectic 4-th","Adaptive RK45"])
    plt.show()

    '''Plotting Trajectories at T = 19000 days'''
    world.SetTAmount(256000)
    fig = plt.figure(2)
    ax1 = fig.add_subplot(111)
    world.CalcTrajectories(19000*T_day,World.fourthsymp)
    world.SetAxisLimits(ax1)
    [o.Draw(T,False,ax1) for o in world.objects]
    plt.show()
