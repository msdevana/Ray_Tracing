"""
MANATEE TOOLBOX (OCEANOGRAPHY TOOLS)

Oceanography tools which cover standard calculations and plots with 
as much freedom to customize plots as possible. 

Xarray is used as the primary tool for handling large datasets but H5PY is also 
used to handle HDF files which are common for some large datasets

Manish Devana, 2018

"""

import numpy as numpy
import xarray as xr 
import h5py 
import gsw
from netCDF4 import Dataset
import datetime as dt



def mat2xr(matfile):
    """
    Convert matlab files into xarray objects which can be stored either as 
    pickled objects or netcdf files. Doing this conversion is tedious and slow 
    so hopefully this helps saves time. 

    Matlab files are loaded using h5py (issues may rise from older forms of 
    matlab files). Nested structures will also give you some issues

    ** CANNOT HANDLE NESTED STRUCTURES RIGHT NOW **

    

    PARAMETERS
    ----------
    matfile: string of matlab file path/name ('.mat') files only


    RETURNS
    -------
    """

    # load matlab file 
    file = h5py.File(matfile)

    # iterate through keys
    keys = list(file.keys())
    dataset = {} # store matlab data in dictionary of xarray dataArrays
    for key in keys:
        dataset[key] = xr.DataArray(file[key][:])

    file.close() # close matlab file after use

    return dataset # return dict of xarray dataArrays
        


def gebco():
    """
    Loads GEBCO bathymetry data into xarrray datasets

    """


def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)

    return day + dayfrac


def profile(values, dim):
    g = 0

def transect(profiles, dims):
    k = 0