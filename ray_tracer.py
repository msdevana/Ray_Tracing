"""
Created By: Manish S. Devana (mdevana@rsmas.miami.edu)

Ray Tracing in 4 dimensions using satGEM background fields

"""


import numpy as np
from scipy.integrate import solve_ivp, RK23, odeint,  quad
from scipy.interpolate import RegularGridInterpolator, interp2d
import xarray as xr
from xarray import ufuncs as xru
import manatee as mt
import matplotlib.pyplot as plt
import oceans as oc
import datetime as dt
import gsw
from ipywidgets import FloatProgress
from IPython.display import display
from numba import jit
import pandas as pd

# Background functions

#-------------------------------------------------------------------------------
class LinearNDInterpolatorExt(object):
    """
    modified regular grid inteprolator
    """

    def __init__(self, points, values, fill_value=None):
        self.funcinterp = RegularGridInterpolator(
            points, values, method='linear', bounds_error=False, fill_value=np.nan)
        self.funcnearest = RegularGridInterpolator(
            points, values, method='nearest', bounds_error=False, fill_value=fill_value)

    def __call__(self, *args):

        t = self.funcinterp(*args)
        if np.isfinite(t):
            return t.item(0)
        else:
            return self.funcnearest(*args)

#-------------------------------------------------------------------------------

def inverse_hav(x, y, lon1, lat1):
    """
    Uses the inverse haversine function to convert x and y distance to a new lat and long coordinate. (see ray tracing docs for full formula)

    Parameters
    ----------
    x: x distance traveled (east-west)
    y: y distance traveled (north-south)
    lon1: starting longitude (Degrees)
    lat1: starting latitude (Degrees)

    Returns
    -------
    lon2: final longitude (Degrees)
    lat2: final latitude (Degrees)

    
    """

    r = 6371e3  # radius of the earth
    d = np.sqrt(x**2 + y**2)  # total distance traveled

    lat2 = lat1 + (y / 111.11e3)  # convert y distance to a new latitude point

    # Convert to radians for use in trig functions

    lon2 = lon1 + (x / r) * (180 / np.pi) / np.cos(lat1 * np.pi / 180)

    return lon2, lat2  # in degrees
#-------------------------------------------------------------------------------

class gemFuncs(object):
    """
    Object that stores the satGEM functions 
    """

    def __init__(self):
        """
        Generate gemFuncs object 
        """

        self.gem = xr.open_dataset('satGEM_md.nc')
        self.bathy_file = xr.open_dataset('bathy.nc')
        trev = []
        for i in range(self.gem.time.shape[0]):
            trev.append(oc.matlab2datetime(
                self.gem.time[i].values).toordinal())

        trev = np.asarray(trev)
        self.gem['time'] = trev

#-------------------------------------------------------------------------------


    def createFuncs(self, X, t, lonpad=1.5, latpad=1.5, tpad=1.5):
        """
        Pass data fields of satgem to generate functions 
        
        """

        """
        Generate Interpolation functions 
        """
        if tpad < 7:
            tpad = 7
        lonind = self.gem.temp.lon.sel(lon=slice(X[0] - lonpad, X[0] + lonpad))
        clonind = self.gem.V.clon.sel(clon=slice(X[0] - lonpad, X[0] + lonpad))
        latind = self.gem.temp.lat.sel(lat=slice(X[1] - latpad, X[1] + latpad))
        clatind = self.gem.U.clat.sel(clat=slice(X[1] - latpad, X[1] + latpad))
        tind = self.gem.temp.time.sel(time=slice(t - tpad, t + tpad))

        blonind = self.bathy_file.lon.sel(
            lon=slice(X[0] - lonpad, X[0] + lonpad))
        blatind = self.bathy_file.lat.sel(
            lat=slice(X[1] - latpad, X[1] + latpad))

        bsub = self.bathy_file.elevation.sel(lon=blonind, lat=blatind)

        setattr(self, 'bathy', LinearNDInterpolatorExt(
            (blonind, blatind), bsub.T, fill_value=None))

        N2 = []
        rho = []
        for i in range(tind.shape[0]):
            SA = gsw.SA_from_SP(self.gem.sal.sel(lon=lonind, lat=latind, time=tind[i]),
                                self.gem.depth[:], X[0], X[1])
            CT = gsw.CT_from_t(SA, self.gem.temp.sel(lon=lonind, lat=latind, time=tind[0]),
                               self.gem.depth[:])
            if i == 0:
                # since the pmid grid will be uniform, only save once
                n2i, pmid = gsw.Nsquared(SA, CT, self.gem.depth, axis=2)
                N2.append(n2i)
                rho.append(gsw.sigma0(SA, CT, ))
            else:
                N2.append(gsw.Nsquared(SA, CT, self.gem.depth, axis=2)[0])
                rho.append(gsw.sigma0(SA, CT, ))

        FN2 = LinearNDInterpolatorExt((self.gem.lon.sel(lon=lonind),
                                       self.gem.lat.sel(
                                           lat=latind), pmid[0, 0, :],
                                       self.gem.time.sel(time=tind)),
                                      np.stack(N2, axis=3))

        rho1 = LinearNDInterpolatorExt((self.gem.lon.sel(lon=lonind),
                                        self.gem.lat.sel(lat=latind), self.gem.depth,
                                        self.gem.time.sel(time=tind)),
                                        np.stack(rho, axis=3))

        setattr(self, 'N2', FN2)
        setattr(self, 'rho', rho1)

        N2 = np.absolute(np.stack(N2, axis=3))
        N2 = np.sqrt(N2)
        # Don't actually need this as a dataArray but its used as one further down and im too lazy to change it
        N2 = xr.DataArray(N2, coords=[self.gem.lon.sel(lon=lonind),
                                      self.gem.lat.sel(
                                          lat=latind), pmid[0, 0, :],
                                      self.gem.time.sel(time=tind)],
                          dims=['lon', 'lat', 'depth', 'time'], name='N2')

        Usub = self.gem.U.sel(lon=lonind, clat=clatind, time=tind)
        Tsub = self.gem.temp.sel(lon=lonind, lat=latind, time=tind)
        Vsub = self.gem.V.sel(clon=clonind, lat=latind, time=tind)

        # space and time Gradients
        delt = Usub.time.diff(dim='time') * 24 * 60 * \
            60  # time delta in seconds

        # U gradients
        dxu = gsw.distance(np.meshgrid(Usub.lon, Usub.clat)[0],
                           np.meshgrid(Usub.lon, Usub.clat)[1],
                           axis=1)

        dxu = np.repeat(np.repeat(dxu.T[:, :, np.newaxis], Usub.shape[2],
                                  axis=2)[:, :, :, np.newaxis],
                        Usub.shape[3], axis=3)

        dyu = gsw.distance(np.meshgrid(Usub.lon, Usub.clat)[0],
                           np.meshgrid(Usub.lon, Usub.clat)[1],
                           axis=0)
        dyu = np.repeat(np.repeat(dyu.T[:, :, np.newaxis], Usub.shape[2],
                                  axis=2)[:, :, :, np.newaxis],
                        Usub.shape[3], axis=3)
        # V gradients
        dxv = gsw.distance(np.meshgrid(Vsub.clon, Vsub.lat)[0],
                           np.meshgrid(Vsub.clon, Vsub.lat)[1],
                           axis=1)

        dxv = np.repeat(np.repeat(dxv.T[:, :, np.newaxis], Usub.shape[2],
                                  axis=2)[:, :, :, np.newaxis],
                        Usub.shape[3], axis=3)

        dyv = gsw.distance(np.meshgrid(Vsub.clon, Vsub.lat)[0],
                           np.meshgrid(Vsub.clon, Vsub.lat)[1],
                           axis=0)
        dyv = np.repeat(np.repeat(dyv.T[:, :, np.newaxis], Usub.shape[2],
                                  axis=2)[:, :, :, np.newaxis],
                        Usub.shape[3], axis=3)
        # N2 gradient
        dxn = gsw.distance(np.meshgrid(N2.lon, N2.lat)[0],
                           np.meshgrid(N2.lon, N2.lat)[1],
                           axis=1)

        dxn = np.repeat(np.repeat(dxn.T[:, :, np.newaxis], N2.shape[2],
                                  axis=2)[:, :, :, np.newaxis],
                        N2.shape[3], axis=3)

        dyn = gsw.distance(np.meshgrid(N2.lon, N2.lat)[0],
                           np.meshgrid(N2.lon, N2.lat)[1],
                           axis=0)
        dyn = np.repeat(np.repeat(dyn.T[:, :, np.newaxis], N2.shape[2],
                                  axis=2)[:, :, :, np.newaxis],
                        N2.shape[3], axis=3)

        dz = np.nanmean(np.diff(Usub.depth))

        # Spatial Gradient revisied grids
        clat = (Usub.clat[:-1] + np.diff(Usub.clat) / 2)
        clon = (Vsub.clon[:-1] + np.diff(Vsub.clon) / 2)
        lat = (N2.lat[:-1] + np.diff(N2.lat) / 2)
        lon = (N2.lon[:-1] + np.diff(N2.lon) / 2)
        time = Usub.time[:-1] + np.diff(Usub.time) / 2
        pmid = pmid[5, 5, :]
        pmidn = pmid[:-1] + np.diff(pmid) / 2

        setattr(self, 'dudx', LinearNDInterpolatorExt((lon, Usub.clat, Usub.depth,
                                                       Usub.time), Usub.diff(dim='lon').values / dxu, fill_value=0))
        setattr(self, 'dudy', LinearNDInterpolatorExt((Usub.lon, clat, Usub.depth,
                                                       Usub.time), Usub.diff(dim='clat').values / dyu, fill_value=0))
        setattr(self, 'dudz', LinearNDInterpolatorExt((Usub.lon, Usub.clat, pmid,
                                                       Usub.time), Usub.diff(dim='depth').values / dz, fill_value=0))

        setattr(self, 'dvdx', LinearNDInterpolatorExt((clon, Vsub.lat, Usub.depth,
                                                       Usub.time), Vsub.diff(dim='clon').values / dxv, fill_value=1343431))
        setattr(self, 'dvdy', LinearNDInterpolatorExt((Vsub.clon, lat, Usub.depth,
                                                       Usub.time), Vsub.diff(dim='lat').values / dyv, fill_value=0))
        setattr(self, 'dvdz', LinearNDInterpolatorExt((Vsub.clon, Vsub.lat, pmid,
                                                       Vsub.time), Vsub.diff(dim='depth').values / dz, fill_value=0))

        setattr(self, 'dndx', LinearNDInterpolatorExt((lon, N2.lat, N2.depth,
                                                       N2.time), N2.diff(dim='lon').values / dxn, fill_value=0))
        setattr(self, 'dndy', LinearNDInterpolatorExt((N2.lon, lat, N2.depth,
                                                       N2.time), N2.diff(dim='lat').values / dyn, fill_value=0))
        setattr(self, 'dndz', LinearNDInterpolatorExt((N2.lon, N2.lat, pmidn,
                                                       N2.time), N2.diff(dim='depth').values / dz, fill_value=0))

        # Time Gradients Final
        delt = Usub.time.diff(dim='time') * 24 * 60 * \
            60  # time delta in seconds
        dudt = []
        dvdt = []
        dn2dt = []
        for i, dt1 in enumerate(delt):
            dudt.append(Usub.diff(dim='time')[:, :, :, i] / dt1)
            dvdt.append(Vsub.diff(dim='time')[:, :, :, i] / dt1)
            dn2dt.append(N2.diff(dim='time')[:, :, :, i] / dt1)

        setattr(self, 'dndt', LinearNDInterpolatorExt((N2.lon, N2.lat, N2.depth,
                                                       time), np.stack(dn2dt, axis=3), fill_value=0))

        setattr(self, 'dudt', LinearNDInterpolatorExt(
            (Usub.lon, Usub.clat, Usub.depth, time), np.stack(dudt, axis=3), fill_value=0))

        setattr(self, 'dvdt', LinearNDInterpolatorExt(
            (Vsub.clon, Vsub.lat, Vsub.depth, time), np.stack(dvdt, axis=3), fill_value=0))

        setattr(self, 'U', LinearNDInterpolatorExt(
            (Usub.lon, Usub.clat, Usub.depth, Usub.time),
            Usub.values))

        setattr(self, 'V', LinearNDInterpolatorExt(
            (Vsub.clon, Vsub.lat, Vsub.depth, Vsub.time),
            Vsub.values))
        
        setattr(self, 'T', LinearNDInterpolatorExt(
            (Tsub.lon, Tsub.lat, Tsub.depth, Tsub.time),
            Tsub.values))

        lonlim = [N2.lon.min(), N2.lon.max()]
        latlim = [N2.lat.min(), N2.lat.max()]
        tlim = [N2.time.min(), N2.time.max()]

        return lonlim, latlim, tlim

#-------------------------------------------------------------------------------


    def fallback(self, xi):
        """
        fall back in case 
        """
        lonid = self.gem.lon.sel(lon=xi[0], method='nearest').values
        latid = self.gem.lat.sel(lat=xi[1], method='nearest').values
        clonid = self.gem.clon.sel(clon=xi[0], method='nearest').values
        clatid = self.gem.clat.sel(clat=xi[1], method='nearest').values
        zid = self.gem.depth.sel(depth=xi[2], method='nearest').values
        tid = self.gem.time.sel(time=xi[3], method='nearest').values

        xi = (lonid, latid, zid, tid)
        xiu = (lonid, clatid, zid, tid)
        xiv = (clonid, latid, zid, tid)

        return xi, xiu, xiv

    def getfield(self, xi):
        """
        Get field values from satgem
        """
        N2 = self.N2(xi)
        if np.isnan(N2):
            xi1 = self.fallback(xi)[0]
            N2 = self.N2(xi1)

        U = self.U(xi)
        if np.isnan(U):
            xi1 = self.fallback(xi)[1]
            U = self.U(xi1)

        V = self.V(xi)
        if np.isnan(V):
            xi1 = self.fallback(xi)[2]
            V = self.V(xi1)

        dudx = self.dudx(xi)
        if np.isnan(dudx):
            xi1 = self.fallback(xi)[1]
            dudx = self.dudx(xi1)

        dvdx = self.dvdx(xi)
        if np.isnan(dvdx):
            xi1 = self.fallback(xi)[2]
            dvdx = self.dvdx(xi1)

        dndx = self.dndx(xi)
        if np.isnan(dndx):
            xi1 = self.fallback(xi)[0]
            dndx = self.dndx(xi1)

        dudy = self.dudy(xi)
        if np.isnan(dudy):
            xi1 = self.fallback(xi)[1]
            dudy = self.dudy(xi1)

        dvdy = self.dvdy(xi)
        if np.isnan(dvdy):
            xi1 = self.fallback(xi)[2]
            dvdy = self.dvdy(xi1)

        dndy = self.dndy(xi)
        if np.isnan(dndy):
            xi1 = self.fallback(xi)[0]
            dndy = self.dndy(xi1)

        dudz = self.dudz(xi)
        if np.isnan(dudz):
            xi1 = self.fallback(xi)[1]
            dudz = self.dudz(xi1)

        dvdz = self.dvdz(xi)
        if np.isnan(dvdz):
            xi1 = self.fallback(xi)[2]
            dvdz = self.dvdz(xi1)

        dndz = self.dndz(xi)
        if np.isnan(dndz):
            xi1 = self.fallback(xi)[0]
            dndz = self.dndz(xi1)

        return [N2, U, V, dudx, dvdx, dndx, dudy,
                dvdy, dndy, dudz, dvdz, dndz]

#-------------------------------------------------------------------------------
class raytracer(object):
    """
    Integrating ray equations
    
    
    
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
        
        self.F = gemFuncs()
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
   
            u0 = (self.p0 * (k * omega + l * f * 1j) / (omega**2 - f**2))
            v0 = (self.p0 * (l * omega - k * f * 1j) / (omega**2 - f**2))
            b0 = (self.p0 * (-1j * m * field[0]) / (field[0] - omega**2))
            # w0 = self.p0 * (K1[2] * K1[3]) / (field[0] - K1[3]**2)
            # theta = np.arctan2(np.sqrt(K1[0]**2 + K1[1]**2), K1[2])

            # u0 = -np.tan(theta * w0)
            # v0 = -(gsw.f(lat3) / K1[3]) * np.tan(theta * w0)

            # kh = np.sqrt(K1[0]**2 + K1[1]**2)
            # xrev = np.sqrt(dist_so_far[ii,0]**2 + dist_so_far[ii,1]**2)
            xx = np.copy(dist_so_far[ii, 0])
            yy = np.copy(dist_so_far[ii, 1])
            zz = np.copy(dist_so_far[ii, 2])
            phase = k * xx + l * yy \
                    + m * zz - omega * t1

            # step 2
            # w = np.real(w0 * np.exp(1j * phase))
            # # u = np.real(u0 * np.exp(1j * phase))
            # v = np.real(v0 * np.exp(1j * phase))
            # b = np.real(b0 * np.exp(1j * phase))
            

            #step 2 version 2 

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


            Umom = rho0 * (u * w)/period
            Vmom = rho0 * (v * w) / period

            mFlux = np.sqrt(((u * w) / period)**2 + ((v * w) / period)**2)

            # b = -(field[0] /omega / 9.8) * rho0 * w0 * np.sin(phase)

            E = .5 * rho0 * (u2 + v2 + w2) \
                + .5 *rho0* b2 * np.sqrt(field[0])**-2
            # E =E/rho0
 
            energy.append([E, Umom, Vmom, mFlux])

            # allbottom.append(bottom)
            # if K1[3]**2 >= self.F.N2(xi2):
            #     print('frequency above Bouyancy frequency')
            #     print(K[3]**2)
            #     print(self.F.N2(xi2))
            #     break


            if stops:
                if vertspeed:
                    if np.abs(dz2 / tstep) < 1e-4:
                        print('Vertical Group speed = zero {} meters from bottom'.format(bottom- X[2]))
                        break
                    if np.abs(E)> 1000:
                        print('ENERGY ERROR')
                        break

                    if ii > 3:

                        if np.abs(E - energy[ii - 2][0]) > .8 * E:
                            print('Non Linear')
                            break

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

                if X[2] + clearance*np.abs((2*np.pi)/K1[2])  >= bottom:
                    print('Hit Bottom - {} meters from bottom'.format(bottom- X[2]))
                    break

                # if X[2] + 250 >= bottom:
                #     print('Hit Bottom - {} meters from bottom'.format(bottom- X[2]))
                #     break


                # X direction
                # if (2*np.pi)/np.abs(K1[0]) < 50:
                #     print('Horizontal Wavelength (x) approached zero') 
                #     break
                    
                # if (2*np.pi)/np.abs(K1[0]) > 1000e3:
                #     print('Horizontal Wavelength (x) approached infinity')
                #     break
                
                # # Y direction 
                # if (2*np.pi)/np.abs(K1[1]) < 50:
                #     print('Horizontal Wavelength (y) approached zero') 
                #     break
                    
                # if (2*np.pi)/np.abs(K1[1]) > 1000e3:
                #     print('Horizontal Wavelength (y) approached infinity')
                #     break
                
                # # Z direction
                # if (2*np.pi)/np.abs(K1[2]) < 50:
                #     print('Vertical Wavelength (z) approached zero') 
                #     break
                    
                # if (2*np.pi)/np.abs(K1[2]) > 3e3:
                #     print('Vertical Wavelength (z) approached infinity')
                #     break

                if X[2] <= 0:
                    print('Hit Surface')
                    break

                # if K1[3]**2 <= f**2:
                #     print('Fell Below intertial frequency')
                #     print(K[3]**2)
                #     break

                if K1[3]**2 >= self.F.N2(xi2):
                    print('frequency above Bouyancy frequency')
                    # print(K[3]**2)
                    # print(self.F.N2(xi2))
                    break

                if np.iscomplex(K1[0]):
                    print('complex step {}'.format(ii))
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
                                            
                                            

    
        
