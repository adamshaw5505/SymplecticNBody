from scipy.integrate import odeint
from scipy.signal import find_peaks
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class Object:
    def __init__(self, pos, vel, mass, colour=None):
        # Set initial values 
        self.q0 = np.array([pos]); self.v0 = np.array([vel])
        self.qs = np.array([pos])
        self.vs = np.array([vel])
        self.ts = np.array([0])
        self.period = 0
        self.m = mass

        #Give a random colour and pre-define drawing varaibles
        self.c = np.random.rand(3) if colour is None else colour

        self.__line = None
        self.__draw_index = 1000
    
    def SetAxis(self, ax, d3d):
        # Set the axis for this object to draw to
        traj = self.trajectories()
        if d3d:
            self.__line = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=self.c, marker="o",
                                  markevery=slice(-1, -1, 1), markersize=int(10*np.math.log10(self.m/1e29+1)+5))[0]
        else:
            self.__line = ax.plot(traj[:2, 0], traj[:2, 1], color=self.c, marker="o", markevery=slice(
                -1, -1, 1), markersize=int(10*np.math.log10(self.m/1e29+1)+5))[0]

    def Update(self, q, v, t):
        # Adds on a new trajectory point
        self.qs = np.concatenate((self.qs, q.reshape((1, 3))))
        self.vs = np.concatenate((self.vs, v.reshape((1, 3))))
        self.ts = np.append(self.ts, t)

    def Set(self, qs, vs, ts):
        # Sets the entire trajectory ( if its calculated in one go, odeint)
        self.qs = np.asarray(qs)
        self.vs = np.asarray(vs)
        self.ts = np.asarray(ts)

    def __index(self, t=-1):
        # Returns the time index of the time closest to t in self.ts
        t_i = np.argmin(np.abs(self.ts-t))
        return max(0, min(t_i, self.ts.size-1))
        
    def Reset(self):
        # Resets back to initial conditions
        self.qs = self.q0
        self.vs = self.v0
        self.ts = np.array([0])
        self.__draw_index = 1000
        self.period = 0

    def q(self, i=-1, t=-1):
        # Gets position based on index or time
        t_i = i
        if t >= 0 and i == -1: t_i = self.__index(t)
        return self.qs[t_i, :]

    def trajectories(self, i=-1, t=-1):
        # Get trajectories up to a index or time
        t_i = i
        if t >= 0 and i == -1: t_i = self.__index(t)
        return self.qs[0:t_i, :]

    def v(self, i=-1, t=-1):
        # Get velocity at a certain index or time
        t_i = i
        if t >= 0 and i == -1: t_i = self.__index(t)
        return self.vs[t_i, :]

    def Draw(self, time, d3d, axis = None):
        # Draws the object at a specific time on a 3D or 2D axis.
        traj = self.trajectories(t=time)
        if traj.shape[0] == 1: traj = np.repeat(traj, 2, axis=0)
        slc = slice(-self.__draw_index,traj.size,1)
        if axis is not None:
            self.SetAxis(axis,d3d)
        self.__line.set_markevery(slice(-2,-1,1))
        if d3d:
            self.__line.set_data_3d(traj[slc, 0], traj[slc, 1], traj[slc, 2])
        else:
            self.__line.set_data(traj[slc, 0], traj[slc, 1])

    def CalcPeriods(self):
        #This finds when the particles are closest in position to the initial conditions and gets the indices
        lqs = np.linalg.norm(self.qs - self.qs[0,:],axis = 1)
        tqi = find_peaks(np.max(lqs) - lqs)[0]

        #This finds when the particles are closest in velocity to the initial conditions and gets the indices
        vs1 = self.vs / \
            np.linalg.norm(self.vs, axis=1).reshape((self.vs.shape[0], 1))
        tvi = find_peaks(np.abs(np.sum(np.multiply(vs1, vs1[0, :]), axis=1)))[0]

        # This finds the minimum index out of the that appears in both lists of indices and then sets the period to that
        mn = 100
        ti = 0
        for qi in tqi:
            di = np.abs(tvi-qi)
            min_i = np.argmin(di)
            if di[min_i] < mn:
                ti = tvi[min_i]
                mn = di[min_i]
        self.__draw_index = ti
        self.period = self.ts[ti]

class World:
    # Defines the different methods we can use for integration
    fourthsymp = "4THSYMP"
    rk45 = "RK45"
    rk4 = "RK4"

    def __init__(self):
        # Sets everything to empty ready to use
        self.__G = 1
        self.__r_max = self.__nobjs = 0
        self.__times = []
        self.__energies = []
        self.__angular = []
        self.objects = []
        self.__masses = []
        self.__sol_qd = []
        self.__sol_vd = []
        self.__close = False
        self.__pause = True

        self.__stepMethod = self.__symplecticIntegratorStep
        self.__coeffs = None

        self.__t_current = self.__t_old = self.__t_sim = 0.0
        self.__t_new = time.time()
        self.__t_amount = 8000

        self.__dqs = np.array([])
        self.__qss = np.array([])

    #Sets G (the gravitational constant)
    def SetG(self, G): self.__G = G

    # Sets the number of steps we want to use
    def SetTAmount(self, T): self.__t_amount = T

    # Sets the limits of the axis ax to encapsule the objects
    def SetAxisLimits(self,ax, d3d=False):
        if d3d:
            ax.set_xlim3d([-self.__r_max*2, self.__r_max*2])
            ax.set_ylim3d([-self.__r_max*2, self.__r_max*2])
            ax.set_zlim3d([-self.__r_max*2, self.__r_max*2])
        else:
            ax.set_xlim([-self.__r_max*2, self.__r_max*2])
            ax.set_ylim([-self.__r_max*2, self.__r_max*2])

    # Adds a list of objects
    def Add(self, objs):
        self.objects = objs
        self.__nobjs = len(objs)
        for obj in objs:
            self.__masses.append(obj.m)
            self.__r_max = max(np.linalg.norm(obj.q()), self.__r_max)
        # These arrays are for storing the values of constants  and vectors in the acceleration calculation
        # They help save time by not needing to do N^2 calculations
        self.__dqs = [[0 if i == j else -1 for i in range(self.__nobjs)] for j in range(self.__nobjs)]
        self.__qss = [[[0 for i in range(3)] for j in range(self.__nobjs)] for k in range(self.__nobjs)]
        self.__fss = [[0 for i in range(3)] for j in range(self.__nobjs)]


    def __Accels(self, qs):
        #Copies the lists
        fs = [list(x) for x in self.__fss]
        ks = [list(x) for x in self.__dqs]
        _qss = [[list(x) for x in v] for v in self.__qss]
        # Loops through all particles twice
        for i in range(len(qs)):
            for j in range(len(qs)):
                # Checks if the calculation has already been done, if not it then
                # calculates the acceleration.
                if ks[i][j] == -1:
                    _qss[i][j][0] = qs[j][0] - qs[i][0]
                    _qss[i][j][1] = qs[j][1] - qs[i][1]
                    _qss[i][j][2] = qs[j][2] - qs[i][2]

                    _qss[j][i][0] = -_qss[i][j][0]
                    _qss[j][i][1] = -_qss[i][j][1]
                    _qss[j][i][2] = -_qss[i][j][2]
                    k = (_qss[j][i][0]**2 + _qss[j][i][1]**2 + _qss[j][i][2]**2)**(-3/2)
                    ks[i][j] = self.objects[j].m * k
                    ks[j][i] = self.objects[i].m * k

                #Add together all the accelerations.
                fs[i][0] += _qss[i][j][0]*ks[i][j]
                fs[i][1] += _qss[i][j][1]*ks[i][j]
                fs[i][2] += _qss[i][j][2]*ks[i][j]
        #Scale by the graviational constant and return
        return self.__G * np.array(fs)
        
    def __odeFunc(self, y, t):
        # This is the function used by scipy's odeint to integrate with, it's just using the __Accels function
        # but putting it in there format.
        y0 = y.copy()
        y0 = y0.reshape((self.__nobjs*2, 3))
        an = self.__Accels(y0[0::2])
        y0[0::2] = y0[1::2]
        y0[1::2] = an
        return y0.ravel()

    def __symplecticIntegratorStep(self, qs, vs, dt):
        # This is one step of the symplectic integrator, where the coefficients are the 4-th order ones.
        for ai, bi in self.__coeffs:
            an = self.__Accels(qs)
            vs += bi*dt*an
            qs += ai*dt*vs
            self.__t_sim += ai*dt
        return qs, vs

    def __rungeKuttaStep(self, qs, vs, dt):
        # This is the RK45 step and it returns a weighted average of the k values
        kv_1 = self.__Accels(qs)
        kr_1 = vs.copy()
        kv_2 = self.__Accels(qs + kr_1*dt*0.5)
        kr_2 = vs + kv_1 * dt * 0.5
        kv_3 = self.__Accels(qs + kr_2*dt*0.5)
        kr_3 = vs + kv_2 * dt * 0.5
        kv_4 = self.__Accels(qs + kr_3*dt*0.5)
        kr_4 = vs + kv_3 * dt
        qn = qs + (dt/6.0) * (kr_1 + 2*kr_2 + 2*kr_3 + kr_4)
        vn = vs + (dt/6.0) * (kv_1 + 2*kv_2 + 2*kv_3 + kv_4)
        return qn, vn

    def __Integrator(self, qs, vs, times):
        # This is the main integrator functtion, first it splits the times, then sets up the trajectory arrays.
        dts = np.diff(times)
        qs1 = np.zeros((qs.shape[0],qs.shape[1],times.size))
        qs1[:,:,0] = qs
        vs1 = np.zeros((vs.shape[0],vs.shape[1],times.size))
        vs1[:,:,0] = vs

        # This loops through all the times and calculates each step for it
        for i in range(dts.size):
            qs, vs = self.__stepMethod(qs, vs, dts[i])
            qs1[:,:,i+1] = qs
            vs1[:,:,i+1] = vs
        # This sets the solutions to their pre-defined variables.
        self.__sol_qd = qs1
        self.__sol_vd = vs1
        # Then this sets the trajectories and calculates the periods for each object
        for j in range(self.__nobjs):
            self.objects[j].Set(qs1[j,:,:].T,vs1[j,:,:].T,self.__times)
            self.objects[j].CalcPeriods()        

    
    def CalcTrajectories(self, time, method="4THSYMP"):
        # This is the main function to calculate the trajectories. First it resets all the particles
        # and the previous solutions if there are some.
        # Then it sets the integration method based on the one input.
        [o.Reset() for o in self.objects]
        self.__energies = []
        self.__angular = []
        self.__sol_qd = []
        self.__sol_vd = []
        if self.__nobjs == 0: return
        self.__times = np.linspace(0, time, self.__t_amount)
        self.__energies = np.zeros_like(self.__times, dtype = np.float64)
        self.__angulars = np.zeros_like(self.__times, dtype = np.float64)
        qs = np.zeros((self.__nobjs, self.objects[0].q().size))
        vs = np.zeros((self.__nobjs, self.objects[0].v().size))
        for i in range(self.__nobjs):
            qs[i, :] = self.objects[i].q().copy()
            vs[i, :] = self.objects[i].v().copy()
            #This is the fourth order symplectic method
        if method == "4THSYMP":
            c = np.math.pow(2.0, 1.0/3.0)
            self.__coeffs = np.array([[0.5, 0.5*(1.0-c), 0.5*(1.0-c), 0.5],
                               [0.0,         1.0,          -c, 1.0]]).T / (2.0 - c)
            self.__Integrator(
                qs, vs, self.__times)
            # This is Runge-Kutta 4-th order
        elif method == "RK4":
            self.__stepMethod = self.__rungeKuttaStep
            self.__Integrator(qs, vs, self.__times)
            # This is Runge-Kutta 5(4) this ones uses scipy's odeint with 1e-9 and 1e-10 tolerances
        elif method == "RK45":
            y0 = np.zeros(qs.size * 2)
            #Converts the data into the form required for odeint
            for i in range(0, 3):
                y0[i::6] = qs[:, i].ravel()
                y0[3+i::6] = vs[:, i].ravel()
            sol = odeint(self.__odeFunc, y0, self.__times,atol=1e-9, rtol=1e-10, hmax = self.__times[1]) 
            # Then reshapes the data back into the form I use.
            sol2 = sol.reshape(sol.shape[0],self.__nobjs*2,3)
            sol2 = np.swapaxes(sol2,0,2)
            sol2 = np.swapaxes(sol2,0,1)
            self.__sol_qd = sol2[0::2,:,:]
            self.__sol_vd = sol2[1::2,:,:]
            # Then sets the trajectories and calculates the periods for each particle
            for i in range(self.__nobjs):
                self.objects[i].Set(sol[:, 6*i:6*i+3], sol[:, 6*i+3:6*i+6], self.__times)
                self.objects[i].CalcPeriods()
        # Then it does all the energy calculations for the trajectories
        self.CalcEnergies()

    def CalcEnergies(self):
        if self.__nobjs == 0: return
        #Creates an array of the masses to use
        self.__masses = np.array(self.__masses)
        # Calculates the kenetic energy for every particle at all the times.
        self.__energies = np.sum(0.5 * self.__masses.reshape((self.__nobjs,1)) * np.sum(self.__sol_vd**2,axis = 1),axis = 0)
        # Calculates the total angular momentum for every particle at all the times.
        self.__angulars = np.linalg.norm(np.sum(np.cross(self.__sol_vd * self.__masses.reshape((self.__masses.size,1,1)), self.__sol_qd, axis = 1),axis=0),axis = 0)

        # This loop calculates the potential energy for all the particles at all the times then adds it onto the kenetic energies
        # to get the total energy at each point in time for the system.
        for i in np.arange(0,self.__nobjs,1):
            if i == self.__nobjs-1: continue
            q_i = self.__sol_qd[i].reshape(1,self.__sol_qd.shape[1],self.__sol_qd.shape[2])
            dqs = np.linalg.norm(self.__sol_qd[i+1:]-q_i,axis = 1)**(-1)
            if dqs.size > 1:
                dqs =self.__masses[i+1:].reshape((self.__nobjs-i-1,1)) * self.objects[i].m * self.__G * dqs
                self.__energies -= np.sum(dqs,axis = 0)
        
        # This normalises the renergy and angular momentum results so the graphs always show a percentage of the initial
        # energy instead of the actual energy.
        self.__energies = (self.__energies - self.__energies[0])/self.__energies[0]
        if self.__angulars[0] == 0.0: self.__angulars += 1.0
        self.__angulars = (self.__angulars - self.__angulars[0])/self.__angulars[0]        

    def Energy(self, time = None): # Returns energies up to given time, unless time is None then it returns them all
        if time is None: return self.__energies
        return self.__energies[0:np.argmin(np.abs(self.__times - time))]

    def Momentum(self, time):# Returns momenta up to given time, unless time is None then it returns them all
        return self.__angulars[0:np.argmin(np.abs(self.__times - time))]

    @property
    def Times(self): return self.__times # Get the list of calculation times

    @property
    def Periods(self): return [o.period for o in self.objects]  # Returns the periods of all the objects