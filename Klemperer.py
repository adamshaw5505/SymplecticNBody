from nbody_sym import World, Object
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    objects = []
    ''' 5 Body Rosette '''
    T = 31
    TAmount = 64000
    omega = 1.17319304484
    for n in range(0,5):
        theta = (2*np.pi*n)/5
        c_t = np.math.cos(theta)
        s_t = np.math.sin(theta)
        objects.append(Object([c_t,s_t,0],[-omega*s_t,omega*c_t,0],1))

    world = World()
    world.SetG(1)
    world.SetTAmount(TAmount)
    world.Add(objects)

    '''Plotting Trajectories'''
    fig = plt.figure(2)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    world.CalcTrajectories(25,World.rk45)
    [o.Draw(25,False,ax1) for o in world.objects]
    world.CalcTrajectories(31, World.rk45)
    [o.Draw(31,False,ax2) for o in world.objects]
    plt.show()

    '''Comparing Energies Of Different Methods'''
    energy_methods = []
    energy_methods.append(("Symplectic-4th", World.fourthsymp))
    energy_methods.append(("Adaptive RK45", World.rk45))
    energy_list = []
    legend_list = []
    period_list = []
    print("Δt: {0:g}".format(T/TAmount))
    for l,m in energy_methods:
        print("Calculating: {0:}".format(l))
        world.CalcTrajectories(T,m)
        e = world.Energy()
        print("ΔE/E0: ", np.abs(np.max(e)-np.min(e)))
        print("Period: ", np.mean(np.asarray(world.Periods)))
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

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    for e_range in energy_list[:-1]:
        ax.plot(t_range,e_range)
    ax.legend(legend_list)
    plt.ylabel("ΔE/E0")
    plt.xlabel("t")
    plt.show()