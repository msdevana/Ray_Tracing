"""
Created By: Manish S. Devana (mdevana@rsmas.miami.edu)

Ray Tracing in 4 dimensions using satGEM background fields


INSTRUCTIONS FOR RAY TRACING

1 - Load Ray Tracer as object:
    Interpolation functions will be generated when during this step
2 - Set Ray Tracer object wave properties, location, and time
3 - Run raytracing  with specified duration (hours) and timestep(seconds)
4 - results are return as pandas dataframe (can be easily saved as csv)

"""


import numpy as np
from scipy.integrate import solve_ivp, RK23, odeint,  quad
from scipy.interpolate import RegularGridInterpolator, interp2d
import xarray as xr
from xarray import ufuncs as xru
import datetime as dt
import gsw
from ipywidgets import FloatProgress
from IPython.display import display
from numba import jit
import pandas as pd
from core_funcs import *

#-------------------------------------------------------------------------------
class raytracer(object):
    """
    Integrating ray equations
    
    Instructions:
    
    1. Create raytracer object with initial position vector (lon, lat, depth), initial wavenumber vector (lon, lat, depth),
    initial time (in format: ) and longitude /latitude /time padding (these define how big the region for making interpolation functions
    2. use "run" method (i.e. raytracer.run()) to run ray tracing and store results in pandas dataframe format
    
    Parameters:

    X: Position vector (lon, lat, depth)
    K: Wavenumber vector (k, l, m)
    t0: initial time in pandas datetime format
    lonpad:
    latpad:

    
    """

    def __init__(self, X, K, t0, lonpad=1.5, latpad=1.5, tpad=7):
        """
        X = [lon, lat, z]
        K = [k, l , m]
        Keep Initial Conditions
        
        """
        self.X0 = X[:]
        self.K0 = K[:]
        self.t0 = t0
        self.p0 = .02 # random number must be assigned
        
        self.F = gemFuncs() # generate gem functions
        self.lonlim, self.latlim, self.tlim = self.F.createFuncs(
            X, t0, lonpad=lonpad, latpad=latpad, tpad=tpad)

    @jit
    def _cgx(self, N2, Omega, K, u, f):
        return ((K[0] * K[2]**2 * (N2 - f**2)) /
                ((K[0]**2 + K[1]**2 + K[2]**2)**2 * Omega)) + u

    @jit
    def _cgy(self, N2, Omega, K, v, f):
        return ((K[1] * K[2]**2 * (N2 - f**2)) /
                ((K[0]**2 + K[1]**2 + K[2]**2)**2 * Omega)) + v

    @jit
    def _cgz(self, N2, Omega, K, f):
        return (K[0]**2 + K[1]**2) * -K[2] * (N2 - f**2) / ((K[0]**2 + K[1]**2 + K[2]**2)**2 * Omega)

    @jit
    def _dKdt(self, field, K, xi, xi2, tstep):

        K2 = K[0]**2 + K[1]**2 + K[2]**2
        ri = 1 * (np.sqrt(field[0]) * (K[0]**2 + K[1]**2)) / (K2 * K[3])

        dk = (-1 * ri * field[5] - K[0] * field[3] - K[1] * field[4]) * (tstep)
        dl = (-1 * ri * field[8] - K[0] * field[6] - K[1] * field[7]) * (tstep)
        dm = (-1 * ri * field[11] - K[0] * field[9] - K[1] * field[10])*(tstep)

        dudt = (field[1] - self.F.U(xi))/tstep
        dvdt = (field[2] - self.F.V(xi))/tstep
        dndt = (np.sqrt(field[0]) - np.sqrt(self.F.N2(xi)))/tstep



        # dw0 = -1 * ( K[0] * dudt + K[1] * dvdt)
        dw0 = (1* (ri * (dndt) + K[0] * dudt  + K[1] * dvdt))*tstep
        # print(xi)
        # print(xi2)
        # print(dw0)

        return np.array([dk, dl, dm, dw0])

    def _planewave(self,t, amp, xx, yy, zz, k, l, m, omega): 

        return np.real((amp * \
                np.exp((k*xx + l*yy + m*zz - omega*t) * -1j)))**2

    def run(self, tstep=30, duration=5, lonpad=1.5, latpad=1.5,
            tpad=7, direction='forward', bottom=3000,rho0=1030, 
            clearance =.5, shear=-.001, fname='ray_trace.csv',
            strain=True,stops=True,vertspeed=True, 
            time_constant=False, save_data=False, progress_bar=False):
        """
        INSTRUCTIONS:
        TSTEP: TIMESTEP IN SECONDS (DEFAULT 30 SECONDS)
        DURATION: DURATION (IN DAYS) - DEFAULT 5
        IGNORE LONPAD AND LATPAD (DIDNT CHANGE FROM OLDER VERSION)
        DIRECTION: "forward" and "reverse"  (SETS INTEGRATION TIME DIRECTION)
        BOTTOM:  can set default bottom instead of using bathymetry file
        
        Setup for midpoint method integration:
        1. Get field values 
        2. Get Cg @ t_n, X_n
        3. Get field values at t_n+dt/2, X_n + (dt/2)(Cg_n)
        4. Calculate Cg  @(t_n+dt/2, X_n + (dt/2)(Cg_n))
        5. X_(n+1) = X_n +  [dt * Cg  @(t_n+dt/2, X_n + (dt/2)(Cg_n))]
        
        """

        if direction == 'forward':
            # convert duration in hours to seconds
            T = np.arange(0, duration * 60 * 60, tstep)
        else:
            T = np.arange(0, -duration * 60 * 60, -tstep)
            tstep = -tstep

        Xall = []
        Kall = []
        amplitudes = []
        energy = []

        # names of all the columns in results frame
        names = ('Lon', 'Lat', 'depth', 'distance','bottom_depth','k','l','m',
                    'omega','N2', 'U','V', 'dudx', 'dvdx', 'dndx', 'dudy',
                    'dvdy', 'dndy', 'dudz', 'dvdz', 'dndz','cgx','cgy','cgz',
                    'x','y','z', 'u0', 'v0', 'w0', 'u', 'v', 'w', 'b', 'energy',
                    'u_momentum', 'v_momentum', 'horiz_momentum','time' )
        cg = []
        steps = []
        localfield = []
        X = self.X0[:]
        lon0 = X[0]
        lat0 = X[1]
        K = self.K0[:]
        t0 = self.t0
        allbottom = []
        if progress_bar:
            pbar = FloatProgress(min=0, max=T.shape[0])
            pbar.value
            display(pbar)

        if not hasattr(self.F, 'dudx'):
            lonlim, latlim, tlim = self.F.createFuncs(X, lonpad, latpad, tpad)

        for ii, t1 in enumerate(T):
            # Get field values
            if progress_bar:
                pbar.value = float(ii)
            t = t0 + t1 / (24 * 60 * 60)
            if X[2] > 6000:
                zi1 = 2499
            else:
                zi1 = X[2]

            f = gsw.f(X[1])
            if time_constant:
                t = np.copy(t0)
            xi = (X[0], X[1], zi1, t)
            field = self.F.getfield(xi)

            f = gsw.f(X[1])

            # Step 1
            dy1 = self._cgy(field[0], K[3], K, field[2], f) * tstep / 2
            dz1 = self._cgz(field[0], K[3], K, f) * tstep / 2
            dx1 = self._cgx(field[0], K[3], K, field[1], f) * tstep / 2

            # midpoint position

            lon2, lat2 = inverse_hav(dx1, dy1, X[0], X[1])
            if X[2] + dz1 > 6000:
                zi = 2499
            else:
                zi = X[2] + dz1
            xi2 = (lon2, lat2, zi, t + tstep / (24 * 60 * 60 * 2))
            if time_constant:
                xi2 = (lon2, lat2, zi, t)
            field1 = self.F.getfield(xi2)
            f2 = gsw.f(lat2)

            # Update Wave properties at midpoint (midpoint refraction)
            dK = self._dKdt(field1, K, xi, xi2, tstep/2)

            if not np.all(np.isfinite(dK)):

                K1 = K[:]
            else:

                if strain:

                    K1 = [K[0] + dK[0],
                        K[1] + dK[1],
                        K[2] + dK[2],
                        K[3] + dK[3]]
                else:
                     K1 = [K[0] ,
                           K[1] ,
                           K[2] + (tstep/2)*(-(shear)*(K[0] + K[1])) ,
                           K[3] + dK[3]]

            # Step2
            dx2 = self._cgx(field1[0], K1[3], K1, field1[1], f2) * tstep
            dy2 = self._cgy(field1[0], K1[3], K1, field1[2], f2) * tstep
            dz2 = self._cgz(field1[0], K1[3], K1, f2) * tstep

            lon3, lat3 = inverse_hav(dx2, dy2, X[0], X[1])

            lonr = np.expand_dims(np.array([lon0, lon3]), axis=1)
            latr = np.expand_dims(np.array([lat0, lat3]), axis=1)
            distance = gsw.distance(lonr, latr, axis=0)

            if X[2] + dz2 > 6000:
                zi = 2499

            bathypad = np.linspace(-.01,.01, num=5)
            loncheck = bathypad + X[0]
            latcheck = bathypad + X[1]
            loncheck, latcheck = np.meshgrid(loncheck, latcheck)
            tester = np.array([loncheck.flatten(), latcheck.flatten()])
            bottom = np.nanmax([-self.F.bathy((p1[0], p1[1])) \
                                    for p1 in tester.T])
            # bottom = -self.F.bathy((X[0], X[1]))
            X1 = [lon3, lat3, X[2] + dz2, distance, bottom]


            steps.append([dx2, dy2, -dz2])
            cg.append([dx2 / tstep, dy2 / tstep, -dz2 / tstep])
            localfield.append(field)
            Kall.append(K1)
            K = K1
            Xall.append(X1)
            X = X1

            dist_so_far = np.cumsum(steps, axis=0)
            # print(dK[3])
            # print(K[3]**2)
            k = np.copy(K1[0])
            l = np.copy(K1[1])
            m = np.copy(K1[2])
            omega = np.copy(K1[3])
            f = gsw.f(lat3)
            w0 = (self.p0 * (-m * omega) / (field[0] - omega**2))

            # Perturbation amplitudes
            u0 = (self.p0 * (k * omega + l * f * 1j) / (omega**2 - f**2))
            v0 = (self.p0 * (l * omega - k * f * 1j) / (omega**2 - f**2))
            b0 = (self.p0 * (-1j * m * field[0]) / (field[0] - omega**2))

            # total distance so far
            xx = np.copy(dist_so_far[ii, 0])
            yy = np.copy(dist_so_far[ii, 1])
            zz = np.copy(dist_so_far[ii, 2])
            phase = k * xx + l * yy \
                    + m * zz - omega * t1



            # INtegration Limits
            period = np.abs(2 * np.pi / omega)
            t11 = t1 - period /2
            t22 = t1 + period /2



            # mean value theorem to get average over one wave period
            u2 = .5*np.real(w0)**2
            v2 = .5*np.real(v0)**2
            w2 = .5*np.real(w0)**2
            b2 = .5 * np.real(b0)**2
                         
            u = (quad(self._planewave, t11, t22,
                                 args=(u0, xx, yy, zz, k, l,
                                 m,omega))[0])
            v = (quad(self._planewave, t11, t22,
                                 args=(v0, xx, yy, zz, k, l,
                                 m,omega))[0])
            w = (quad(self._planewave, t11, t22,
                                 args=(w0, xx, yy, zz, k, l,
                                 m,omega))[0])

            b = (quad(self._planewave, t11, t22,
                                 args=(b0, xx, yy, zz, k, l,
                                 m,omega))[0])                                          
            
            amplitudes.append([u0, v0, w0, u, v, w, b])

            # Calculate U and V momentum 
            Umom = rho0 * (u * w)/period
            Vmom = rho0 * (v * w) / period

            # Calculate momentum flux
            mFlux = np.sqrt(((u * w) / period)**2 + ((v * w) / period)**2)

            # b = -(field[0] /omega / 9.8) * rho0 * w0 * np.sin(phase)
            # Internal wave energy
            E = .5 * rho0 * (u2 + v2 + w2) \
                + .5 *rho0* b2 * np.sqrt(field[0])**-2
            # E =E/rho0
 
            energy.append([E, Umom, Vmom, mFlux])


            if stops:
                # check if vertical speed goes to zero
                if vertspeed:
                    if np.abs(dz2 / tstep) < 1e-4:
                        print('Vertical Group speed = zero {} meters from bottom'.format(bottom- X[2]))
                        break
                    if np.abs(E)> 1000:
                        # this checks if energy has gone to some unrealistic asymptote like behavior
                        print('ENERGY ERROR')
                        break

                    if ii > 3:

                        if np.abs(E - energy[ii - 2][0]) > .8 * E:
                            print('Non Linear')
                            break

                # data Boundary checks
                if not self.lonlim[0] <= X[0] <= self.lonlim[1]:
                    print('lon out of bounds')
                    break

                if not self.latlim[0] <= X[1] <= self.latlim[1]:
                    print('lat out of bounds')
                    break

                if not self.tlim[0] <= t <= self.tlim[1]:
                    print('time out of bounds')
                    print(t)
                    print(self.tlim)
                    break

                #  Check if near the bottom or surface
                if X[2] + clearance*np.abs((2*np.pi)/K1[2])  >= bottom:
                    print('Hit Bottom - {} meters from bottom'.format(bottom- X[2]))
                    break


                if X[2] <= 0:
                    print('Hit Surface')
                    break

       
                # Check if  frequency gets too high
                if K1[3]**2 >= self.F.N2(xi2):
                    print('frequency above Bouyancy frequency')
                    # print(K[3]**2)
                    # print(self.F.N2(xi2))
                    break



                if not np.isfinite(X1[0]):
                    print('X Update Error')

                    break

                if np.abs(u0) < 0.0001:
                    print('U amplitude zero')

                    break
                if np.abs(v0) < 0.0001:
                    print('v amplitude zero')

                    break
                if np.abs(w0) < 0.0001:
                    print('w amplitude zero')

                    break

                if not np.isfinite(dx1):
                    print('Field Error')
                
                    break

        # Save data in pandas data 
        data = pd.DataFrame(np.concatenate(
                (np.real(np.stack(Xall)),
                np.real(np.stack(Kall)), 
                np.real(np.stack(localfield)),
                np.real(np.stack(cg)),
                np.real(np.stack(np.cumsum(steps, axis=0))),
                np.real(np.stack(amplitudes)),
                np.real(np.stack(energy)),
                np.real(np.expand_dims(T[:ii + 1], axis=1))), axis=1), columns=names)

        if save_data:
            data.to_csv(fname)

        return data
                                            
                                            

    
        
