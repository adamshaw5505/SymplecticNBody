from nbody import World, Object
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    objects = []

    ''' Figure 8 3-Body Test '''
    
    T = 12
    TAmount = 128000
    # Initial conditions for the Figure 8 3-Body System ( Pos, Vel, mass )
    objects.append(Object([0.97000436, -0.24308753, 0.0],
                     [0.466203685, 0.43236573, 0.0], 1))
    objects.append(Object([-0.97000436, 0.24308753, 0.0],
                     [0.466203685, 0.43236573, 0.0], 1))
    objects.append(Object([0, 0, 0], [-0.93240737, -0.86473146, 0], 1))

    world = World()
    world.SetG(1)
    world.SetTAmount(TAmount)
    world.Add(objects)

    '''Plotting Trajectories'''
    print("Δt: {0:g}".format(T/TAmount))
    world.CalcTrajectories(T,World.rk4)
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    world.SetAxisLimits(ax)
    [o.Draw(T,False,ax) for o in world.objects]
    plt.show()

    '''Comparing Energies Of Different Methods'''
    energy_methods = []
    energy_methods.append(("Symplectic-4th", World.fourthsymp))
    energy_methods.append(("Adaptive RK45", World.rk45))
    energy_methods.append(("RK4", World.rk4))
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

    '''Energy Range Vs Step size'''
    N_range = np.linspace(128000//10,128000*2,100)
    de_range = np.zeros_like(N_range)
    de1_range = np.zeros_like(N_range)
    for i in range(N_range.size):
        world.SetTAmount(N_range[i])
        world.CalcTrajectories(T,World.rk45)
        e = world.Energy()
        de_range[i] = np.abs(np.max(e)-np.min(e))
        world.CalcTrajectories(T,World.fourthsymp)
        e = world.Energy()
        de1_range[i] = np.abs(np.max(e)-np.min(e))
    
    plt.plot(T/N_range,de_range)
    plt.plot(T/N_range,de1_range)
    plt.legend(["RK45","Symplectic"])
    plt.show()
