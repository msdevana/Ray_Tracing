"""
Created by: Manish Devana (2018)

This module stores all the necessary ray equations of 4D ray tracing. 

"""


import numpy as np
import gsw
import oceans as oc


def CGx(N2, Omega, k, l, m, u, f):
    """
    Horizontal group speed in x-direction

    Parameters
    ----------
    N2: Buoyancy Frequency Squared
    Omega: intrinsic wave frequency
    k: zonal wavenumber
    l: meridional wavenumber 
    m: vertical wavenumber
    u: zonal flow speed
    f: coriolis parameter

    Returns
    -------
    cgx: Horizontal (x) Group Speed (m/s)


    """
#    K2 = k**2 + l**2 + m**2

    cgx = ((k * m**2 * (N2 - f**2)) / ((k**2 + l**2 + m**2)**2 * Omega)) + u

    return cgx


def CGy(N2, Omega, k, l, m, v, f):
    """
    Horizontal group speed in y-direction in a flow

    Parameters
    ----------
    N2: Buoyancy Frequency Squared
    Omega: intrinsic wave frequency
    k: zonal wavenumber
    l: meridional wavenumber
    m: vertical wavenumber
    v: meridional flow speed
    f: coriolis parameter

    Returns
    -------
    cgy: Horizontal (y) Group Speed (m/s)

    """

    K2 = k**2 + l**2 + m**2

    cgy = ((l * m**2 * (N2 - f**2)) / ((k**2 + l**2 + m**2)**2 * Omega)) + v

    return cgy

def refraction(N, k, l, m, dNdi, Omega):
    """
    Refraction index of internal wave through stratification

    Parameters
    ----------
    N: Buoyancy Frequency

    k: zonal wavenumber
    l: meridional wavenumber
    m: vertical wavenumber
    u: zonal flow speed
    f: coriolis parameter
    Omega: intrinsic wave frequency

    Returns
    -------
    ri: refraction index for a single direction (x, y, or z)
    """

    K = k**2 + l**2 + m**2
    ri=((N * (k**2 + l**2)) / (K * Omega)) * (dNdi)

    return ri

def del_wavenum(dudi, dvdi, k, l, m, N, dndi, Omega):
    """
    Change of zonal wavenumber k in time

    Parameters
    ----------
    dU: change in U (zonal velocity)
    dV: change in V (meridional velocity)
    dx: x change (meters)
    k: zonal wavenumber
    l: meridional wavenumber
    m: vertical wavenumber
    dN: change in buoyancy frequency
    N: Buoyancy frequency
    Omega: Intrinsic frequency

    Returns
    -------
    dkdt: Change in zonal wavenumber k

    """
    ri=refraction(N, k, l, m, dndi, Omega)

    dkdt = -1 * (ri + k * (dudi) + l * (dvdi))

    return dkdt

def dOmega(rx, ry, rz, k, l, dudt, dvdt):
    """
    Change in intrinsic frequency / dispersion relation

    Paramaters
    ----------
    rx: wave refraction in x direction
    ry: wave refraction in y direction
    rz: wave refraction in z direction
    k: zonal wavenumber
    l: meridional wavenumber
    dU: change in U (zonal velocity)
    dV: change in V (meridional velocity)

    Returns
    -------
    dW: Change in frequency

    """

    dWdt = (rx + ry + rx) + k * dudt + l * dudt

    return dWdt


def datetime2ordinal(time):
    """
    convert datetime to ordinal number with decimals 
    
    
    """

    ordinal_time = time.toordinal() + time.hour / 24 + time.second / (24 * 60 * 60)
    
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
    latrev1 = np.deg2rad(lat1)
    latrev2 = np.deg2rad(lat2)

    # inverse haversine formula
    shift = 0.5 * np.rad2deg(np.arccos(1 - 2 * ((np.sin(d / (2 * r))**2
                                                 - np.sin((latrev2 - latrev1) / 2)**2) /
                                                (np.cos(latrev1) * np.cos(latrev2)))))

    if x < 0:
        lon2 = lon1 - shift
    else:
        lon2 = lon1 + shift

    return lon2, lat2  # in degrees
