"""
The core functions for ray tracing 

"""

import numpy as np
from scipy.integrate import solve_ivp, RK23, odeint,  quad
from scipy.interpolate import RegularGridInterpolator, interp2d
import scipy.interpolate as interp
import xarray as xr
from xarray import ufuncs as xru
import matplotlib.pyplot as plt
import datetime as dt
import gsw
import cmocean
from ipywidgets import FloatProgress
from IPython.display import display
from numba import jit
import pandas as pd



def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)

    return day + dayfrac

#-------------------------------------------------------------------------------
class LinearNDInterpolatorExt(object):
    """
    The modified regular grid interpolator which generates both nearest neighbor and linear N-D interpolator 
    When called the linear interpolation is attempted first and if the result is Nan then it falls back to nearest neighbor interpolation
    """

    def __init__(self, points, values, fill_value=None):
        self.funcinterp = RegularGridInterpolator(
            points, values, method='linear', bounds_error=False, fill_value=np.nan)
        self.funcnearest = RegularGridInterpolator(
            points, values, method='nearest', bounds_error=False, fill_value=fill_value)

    def __call__(self, *args):
        
        # Try Linear Neighbor interpolation First
        t = self.funcinterp(*args)

        if np.isfinite(t):
            return t.item(0)
        else: # If nearest linear interpolation fails then use nearest neighbor
            return self.funcnearest(*args)


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



# ---------------------------------------------------------------
class gemFuncs(object):
    """
    Object that stores the satGEM functions 
    """

    def __init__(self):
        """
        Generate gemFuncs object which loads the satGEM fields and the local bathymetry. This needs to be modified for other locations
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
        Pass data fields of satgem to generate interpolation functions 
    
        Generate Interpolation functions 

        Paramaters
        ----------
        X: Position Vector
        t: initial time (center of the interpolation field)
        """
        if tpad < 7:
            tpad = 7

        # Get indicies for subset of satgem/bathy data
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
        fall back in case interpolation doesn't wokr just use nearest neighbor interpolation
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
        Get field values from satgem interpolation functions
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