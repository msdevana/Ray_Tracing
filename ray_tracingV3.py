"""
Created by: Manish Devana (2018)

Ray tracing of internal gravity waves using satGEM velocity, temperature, and 
salinity fields. 

"""


import numpy as np
import gsw
import oceans as oc
from oceans import LinearNDInterpolatorExt
import matplotlib.dates as mdates
import h5py
from datetime import datetime, timedelta
from netCDF4 import Dataset
from scipy.interpolate import Rbf, LinearNDInterpolator, interp2d, NearestNDInterpolator
from mpl_toolkits.mplot3d import Axes3D
import ray_equations as eq



class Wave(object):
    """
    Stores features of a wave as an object which can be input into
    the ray tracing function

    """

    # Add functionality for a default Buoyancy Frequncy and Velocity Profile

    def __init__(self, k=(2 * np.pi) / 1000, l=(2 * np.pi) / 1000,
                 t0=datetime(2012, 11, 2, 3, 0, 0),
                 m=500, w0=-1.3e-4, z0=500, lat=-55, lon=-55):

        # Convert wavelengths into wavenumbers
        # Save initial values becuase running the model will change
        # the wave features.
        self.k = np.array([k], dtype='float')
        self.l = np.array([l], dtype='float')
        self.m = np.array([m], dtype='float')
        self.w0 = np.array([w0], dtype='float')
        self.kh = np.array([np.sqrt(self.k**2 + self.l**2)])
        self.z0 = np.asscalar(np.array([z0], dtype='float'))
        self.lat0 = np.asscalar(np.array([lat], dtype='float'))
        self.lon0 = np.asscalar(np.array([lon], dtype='float'))
        self.t0 = t0


class satGEM_mini(object):
    """
    --------------------TRIMMED VERSION OF SATGEM---------------------

    load in the satGEM data as an object (this might be wierd though because the h5py module loads in each file as an object so not sure...)
    
    The objects built in functions can then be used to easily access the data set without ever having to load the whole thing in.
    
    Also Contains bathymetry data from GEBCO which is also stored as an object
    rather than loaded in all at once. 
    
    """

    def __init__(self):
        # Load satGEM data as h5py file objects
        file = h5py.File('satGEM_update_md.mat')

        self.u = file['urev']
        self.v = file['vrev']
        self.N2 = file['n2']
        self.N2grid = file['n2grid']

        # Data grids
        time = np.squeeze(np.array(file['trev']))
        # convert from matlab to python date time.
        self.time = np.array([oc.matlab2datetime(timeIn) for timeIn in time])
        self.timevec = np.array([t.toordinal() for t in self.time])

        self.depth_grid = file['depthlvl']
        self.lon = file['lons2'][:].flatten()
        self.lat = file['lats2'][:].flatten()

        # The u and v grids are one point off each so I need
        # to figure out how to handle this
        # u uses centered lat
        self.centerlat = file['clats2'][:].flatten()
        # v uses centered lon
        self.centerlon = file['clons2'][:].flatten()

        # gradients
        self.dudx = file['dudx']
        self.dvdx = file['dvdx']
        self.dndx = file['dndx']

        self.dudy = file['dudy']
        self.dvdy = file['dvdy']
        self.dndy = file['dndy']

        self.dudz = file['dudz']
        self.dvdz = file['dvdz']
        self.dndz = file['dndz']

        self.dudt = file['dudt']
        self.dvdt = file['dvdt']

        # bathymetry data
        self.bathy = Dataset('bathy.nc')

        self.ngrids = [self.N2, self.dndx, self.dndy, self.dndz]
        self.ugrids = [self.u, self.dudx, self.dudy, self.dudz, self.dudt]
        self.vgrids = [self.v, self.dvdx, self.dvdy, self.dvdz, self.dvdt]

    def _locate_variables(self, lon, lat, depth, time,
                          tres=1, xres=10, yres=10, zres=5):
        """
        Locate point/points within the satgem data set

        Parameters
        ----------
        lon: longitude of point
        lat: latitude of point
        depth: depth of point
        time: of point

        Returns
        -------
        lon_id: index along longitude axis
        lat_id: index along latitude axis
        depth_id: index along latitude axis
        time_id: index along time axis

        These are for the velocity grids which are offset
        centerlon_id: index along centerlon axis
        centerlat_id: index along centerlat axis


        """

        # Add warning for out of time and space boundaries.

        # get indices
        lon_id = np.argmin(np.abs(self.lon[:] - lon))
        lat_id = np.argmin(np.abs(self.lat[:] - lat))
        depth_id = np.argmin(np.abs(self.depth_grid[:] - depth))
        time_id = np.argmin(np.abs(self.time[:] - time))

        clon_id = np.argmin(np.abs(self.centerlon[:] - lon))
        clat_id = np.argmin(np.abs(self.centerlat[:] - lat))

        # interpolate to position

        # y interps
        lonpad = 1
        latpad = 1
        tpad = 1

        n2_in = self.N2[lon_id - lonpad:lon_id + lonpad + 1,
                        lat_id - latpad:lat_id + latpad + 1,
                        :, time_id - tpad:time_id + tpad + 1]

        lonvec = self.lon[lon_id - lonpad:lon_id + lonpad + 1]
        latvec = self.lat[lat_id - latpad:lat_id + latpad + 1]
        tvec = self.timevec[time_id - tpad:time_id + tpad + 1]

        return lon_id, lat_id, depth_id, time_id, clon_id, clat_id

    def interpfield_lnd(self, lon, lat, depth, time,
                        lonpad=0.5, latpad=0.5, tpad=1):
        """
        Generate 4D radial basis function for interpolating
        ray tracing points
        
        """

        tpad = timedelta(weeks=tpad)

        # get indices for center of interpolation field
        lon_id1, lat_id1, depth_id1, time_id1, clon_id1, clat_id1 = \
            self._locate_variables(lon - lonpad,
                                   lat - latpad,
                                   depth,
                                   time - tpad)

        lon_id2, lat_id2, depth_id2, time_id2, clon_id2, clat_id2 = \
            self._locate_variables(lon + lonpad,
                                   lat + latpad,
                                   depth,
                                   time + tpad)

        # make subgrid

        lonvec = self.lon[lon_id1:lon_id2 + 1]
        latvec = self.lat[lat_id1:lat_id2 + 1]
        tvec = self.timevec[time_id1:time_id2 + 1]

        latmesh, lonmesh, dmesh, tmesh = np.meshgrid(latvec,
                                                     lonvec,
                                                     self.N2grid[:],
                                                     tvec, sparse=False)

        # order = [regular, dx, dy, dz, dt]
        names = ['Fn2', 'Fdndx', 'Fdndy', 'Fdndz']

        for i, grid in enumerate(self.ngrids):
            subgrid = grid[lon_id1:lon_id2 + 1,
                           lat_id1:lat_id2 + 1,
                           :, time_id1:time_id2 + 1]

            mask = ~np.isnan(subgrid)
            points = np.vstack([lonmesh[mask],
                                latmesh[mask],
                                dmesh[mask],
                                tmesh[mask]]).T

            F = LinearNDInterpolatorExt(points,
                                        subgrid[mask])

            setattr(self, names[i], F)

        # U functions -  u uses centered latitude grid
        latvec = self.centerlat[clat_id1:clat_id2 + 1]
        latmesh, lonmesh, dmesh, tmesh = np.meshgrid(latvec,
                                                     lonvec,
                                                     self.depth_grid[:],
                                                     tvec, sparse=False)

        names = ['Fu', 'Fdudx', 'Fdudy', 'Fdudz', 'Fdudt']
        for i, grid in enumerate(self.ugrids):
            subgrid = grid[lon_id1:lon_id2 + 1,
                           lat_id1:lat_id2 + 1,
                           :, time_id1:time_id2 + 1]

            mask = ~np.isnan(subgrid)
            points = np.vstack([lonmesh[mask],
                                latmesh[mask],
                                dmesh[mask],
                                tmesh[mask]]).T

            F = LinearNDInterpolatorExt(points,
                                        subgrid[mask])

            setattr(self, names[i], F)

        # V functions -  u uses centered latitude grid
        clonvec = self.centerlon[clon_id1:clon_id2 + 1]

        # remake lat vector
        latvec = self.lat[lat_id1:lat_id2 + 1]

        latmesh, lonmesh, dmesh, tmesh = np.meshgrid(latvec,
                                                     clonvec,
                                                     self.depth_grid[:],
                                                     tvec, sparse=False)
        # create rbf functions for each paramater F(lon, lat, depth, time)
        names = ['Fv', 'Fdvdx', 'Fdvdy', 'Fdvdz', 'Fdvdt']
        for i, grid in enumerate(self.vgrids):
            subgrid = grid[lon_id1:lon_id2 + 1,
                           lat_id1:lat_id2 + 1,
                           :, time_id1:time_id2 + 1]

            mask = ~np.isnan(subgrid)
            points = np.vstack([lonmesh[mask],
                                latmesh[mask],
                                dmesh[mask],
                                tmesh[mask]]).T

            F = LinearNDInterpolatorExt(points,
                                        subgrid[mask])

            setattr(self, names[i], F)

        # return functions' boundaries
        latlims = [np.nanmin(latvec), np.nanmax(latvec)]
        lonlims = [np.nanmin(lonvec), np.nanmax(lonvec)]
        tlims = [np.nanmin(tvec), np.nanmax(tvec)]

        subgrid = self.N2[lon_id1:lon_id2 + 1,
                          lat_id1:lat_id2 + 1,
                          :, time_id1:time_id2 + 1]

        # create interpolation function for bathymetry
        lonid1 = np.argmin(np.abs((lon - lonpad) - self.bathy['lon'][:]))
        lonid2 = np.argmin(np.abs((lon + lonpad) - self.bathy['lon'][:]))

        latid1 = np.argmin(np.abs((lat - latpad) - self.bathy['lat'][:]))
        latid2 = np.argmin(np.abs((lat + latpad) - self.bathy['lat'][:]))

        lonvec = self.bathy['lon'][lonid1:lonid2 + 1]
        latvec = self.bathy['lat'][latid1:latid2 + 1]
        bathy_subgrid = self.bathy['elevation'][latid1:latid2 + 1,
                                                lonid1:lonid2 + 1]

        F = interp2d(lonvec, latvec, bathy_subgrid, kind='cubic')

        setattr(self, 'bathyF', F)

        return lonlims, latlims, tlims


def ray_tracing_interp(wave, gem, energy=None, time_direction='forward',
                       duration=24, tstep=10,
                       latpad=1, lonpad=1, tpad=1,
                       extend_bathy=10000,
                       interp_mode='lnd',
                       interp_function='linear'):
    """
    Ray tracing using the satGEM density and velocity fields with
    local interpolation to avoid strange steps in ray property evolutions
    
    Parameters
    ----------
    wave: Wave object
    gem: satGEM (and bathymetry) data loaded as satGEM_mini object
    time_direction: time direction of 
                    ray tracing ('forward' (default) or 'reverse'))
    duration: duration of run (HOURS) 
    tstep: time step (SECONDS)
    latpad: Latitude padding for subgrid (bigger padding = more satGEM data 
            available but slower runtimes and possible memory issues)
    lonpad: Longitude padding for subgrid (bigger padding = more satGEM data 
            available but slower runtimes and possible memory issues)
    tpad: Time padding for subgrid (bigger padding = more satGEM data 
            available but slower runtimes and possible memory issues)
    extend_bathy: distance(METERS) to extend bathymetry data past end of run
    interp_mode: Method for interpolating and extrapolating through satGEM 
                default = 'lnd', 'rbf'
                Ray tracing is very sensitive to the methods with LND (linear interpolation) being the most consistent (other versions) need 
                further working
    
    Returns
    -------
    results: Dictionary of all results 
    
    
    """

    # argument checks
    if not isinstance(wave, Wave):
        raise ValueError('Wave input must be a Wave object')

    if not isinstance(gem, satGEM_mini):
        raise ValueError('satGEM input must be a satGEM field object')

    # get initial values from wave object
    k = wave.k[:]
    l = wave.l[:]
    m = wave.m[:]
    Omega = wave.w0
    lat = wave.lat0
    lon = wave.lon0
    z = wave.z0
    x = float(0)
    y = float(0)

    x_all = []
    y_all = []
    z_all = []
    k_all = []
    l_all = []
    m_all = []
    om_all = []
    lat_all = []
    lon_all = []
    cgx = []
    cgy = []
    cgz = []
    bathy = []
    u_all = []
    v_all = []
    N2_all = []
    rx_all = []
    ry_all = []
    rz_all = []
    dudz_all = []
    dvdz_all = []
    E_total = []
    if energy:
        action = energy / Omega
    m_flux = []


#    N2_all = []
#    N2_grid = []

    # add start values (theres probably a better way to do this)
    x_all.append(x)
    y_all.append(y)
    z_all.append(z)
    k_all.append(k)
    l_all.append(l)
    m_all.append(m)
    om_all.append(Omega)
    lat_all.append(lat)
    lon_all.append(lon)

    start_time = wave.t0

    # Time arrays and start time in satgem field.
    if time_direction == 'reverse':
        end_time = start_time - timedelta(hours=duration)
        tstep = -tstep
    elif time_direction == 'forward':
        end_time = start_time + timedelta(hours=duration)

    else:
        raise ValueError("Invalid time direction \
                            - accepts 'forward' or 'reverse'")

    time = np.arange(start_time, end_time,
                     timedelta(seconds=tstep)).astype(datetime)  # create time vector (seconds)
    N2subgrid = None
    if interp_mode == 'rbf':
        lonlims, latlims, tlims = gem.interpfield_rbf(lon, lat,
                                                      z, time[0],
                                                      function=interp_function)
    elif interp_mode == 'lnd':
        lonlims, latlims, tlims = \
            gem.interpfield_lnd(lon, lat,
                                z, time[0],
                                )

    for i, t in enumerate(time[:-1]):

        f = gsw.f(lat)

        # list with [lon, lat, depth, time, centerlon, centerlat] indices
#        lon_idx, lat_idx, z_idx,\
#            t_idx, clon_idx, clat_idx = satGEM.locate(lon, lat, z, t)

        tnum = eq.datetime2ordinal(t)

        # if depth is greater than 3000 and wave hasnt hit bottomr
        # use 3000 (max range of parameter functions)
        # this is assuming that below 3000m values are constant
        latid = np.argmin(np.abs(lat - gem.lat))
        lonid = np.argmin(np.abs(lon - gem.lon))
        timeid = np.argmin(np.abs(gem.time[:] - t))
        N2closest = gem.N2[lonid, latid, :, timeid]
        gembottom = gem.depth_grid[np.max(np.where(np.isfinite(N2closest)))]
        if z > gembottom:
            z1 = gembottom
        else:
            z1 = z
        u = gem.Fu(lon, lat, z1, tnum)
        dudx = gem.Fdudx(lon, lat, z1, tnum)

        dudy = gem.Fdudy(lon, lat, z1, tnum)
        dudz = gem.Fdudz(lon, lat, z1, tnum)
        dudt = gem.Fdudt(lon, lat, z1, tnum)
        if np.isnan(dudt):
            print('dudt ERROR')
            break

        v = gem.Fv(lon, lat, z1, tnum)
        dvdx = gem.Fdvdx(lon, lat, z1, tnum)
        dvdy = gem.Fdvdy(lon, lat, z1, tnum)
        dvdz = gem.Fdvdz(lon, lat, z1, tnum)
        dvdt = gem.Fdvdt(lon, lat, z1, tnum)
        if np.isnan(dvdt):
            print('dvdt ERROR')
            print(tnum)
            print(lon)
            print(lat)
            print(z1)
            break

        N2 = np.abs(gem.Fn2(lon, lat, z1, tnum))
        dndx = gem.Fdndx(lon, lat, z1, tnum)
        dndy = gem.Fdndy(lon, lat, z1, tnum)
        dndz = gem.Fdndz(lon, lat, z1, tnum)

        if time_direction == 'reverse':
            dudt = -dudt
            dvdt = -dvdt

        # Check 1 (these have to be done before calculations)
        if not np.isfinite(N2):
            print('N2 error')
            x_all.append(x)
            y_all.append(y)
            z_all.append(z)
            k_all.append(k)
            l_all.append(l)
            m_all.append(m)
            om_all.append(Omega)
            lat_all.append(lat)
            lon_all.append(lon)
            cgx.append(dx / tstep)
            cgy.append(dy / tstep)
            cgz.append(dz / tstep)
            bathy.append(bottom)
            break

        if not np.isfinite(u):
            print('u error')
            x_all.append(x)
            y_all.append(y)
            z_all.append(z)
            k_all.append(k)
            l_all.append(l)
            m_all.append(m)
            om_all.append(Omega)
            lat_all.append(lat)
            lon_all.append(lon)
            cgx.append(dx / tstep)
            cgy.append(dy / tstep)
            cgz.append(dz / tstep)
            bathy.append(bottom)
            break

        if not np.isfinite(v):
            print('v error')
            x_all.append(x)
            y_all.append(y)
            z_all.append(z)
            k_all.append(k)
            l_all.append(l)
            m_all.append(m)
            om_all.append(Omega)
            lat_all.append(lat)
            lon_all.append(lon)
            cgx.append(dx / tstep)
            cgy.append(dy / tstep)
            cgz.append(dz / tstep)
            bathy.append(bottom)
            break

        # Finite differencing

        # X step
        dx = tstep * eq.CGx(N2, Omega, k, l, m, u, f)
        if np.isnan(CGx(N2, Omega, k, l, m, u, f)):
            print('dx Error')
            break
        x = x + dx  # use this form instead of x+= because it modifies old values

        # Y step
        dy = tstep * eq.CGy(N2, Omega, k, l, m, v, f)
        if np.isnan(dy):
            print('dy Error')
            break
        y = y + dy

        # Z step
        dz = tstep * eq.CGz(Omega, k, l, m, f, N2)
        z = z + dz
        z = np.asscalar(z)

        # k step
        k = k + eq.del_wavenum(dudx, dvdx, k, l, m,
                               np.sqrt(N2), dndx, Omega) * tstep

        # l step
        l = l + eq.del_wavenum(dudy, dvdy, k, l, m,
                               np.sqrt(N2), dndy, Omega) * tstep

        # m step
        m = m + eq.del_wavenum(dudz, dvdz, k, l, m,
                               np.sqrt(N2), dndz, Omega) * tstep

        # Refraction of internal wave through changing stratification
        rx = eq.refraction(np.sqrt(N2), k, l, m, dndx, Omega)
        if np.isnan(rx):
            print('RX ERROR')
            break 

        ry = eq.refraction(np.sqrt(N2), k, l, m, dndy, Omega)
        if np.isnan(ry):
            print('Ry ERROR')
            break

        rz = eq.refraction(np.sqrt(N2), k, l, m, dndz, Omega)
        if np.isnan(rz):
            print('Rz ERROR')
            break

        Omega = Omega + eq.dOmega(rx, ry, rz, k, l, dudt, dvdt)

        # Update position
        lon2, lat2 = eq.inverse_hav(dx, dy, lon, lat)
        lon = np.asscalar(lon2)
        lat = np.asscalar(lat2)

        # find nearest location in bathymetry grid
        bottom = -gem.bathyF(lon, lat)
#        idx1 = np.argmin(np.abs(lon - gem.bathy['lon'][:]))
#        idx2 = np.argmin(np.abs(lat - gem.bathy['lat'][:]))
#        bottom = -1*gem.bathy['elevation'][idx2, idx1]

        # store data
        x_all.append(x)
        y_all.append(y)
        z_all.append(z)
        k_all.append(k)
        l_all.append(l)
        m_all.append(m)
        om_all.append(Omega)
        lat_all.append(lat)
        lon_all.append(lon)
        cgx.append(dx / tstep)
        cgy.append(dy / tstep)
        cgz.append(dz / tstep)
        bathy.append(bottom)
        u_all.append(u)
        v_all.append(v)
        N2_all.append(N2)
        rx_all.append(rx)
        ry_all.append(ry)
        rz_all.append(rz)
        dudz_all.append(dudz)
        dvdz_all.append(dvdz)

        # Check Parameters before next step
        if z > bottom:
            print('Wave hit seafloor')
            break

        if z < 0:
            print('Wave hit surface')
            break

        if np.abs(Omega) < np.abs(f * 1):
            print('Wave Frequency below inertial Frequency')
            break

        # Boundary checks on rbf functions

        if lon <= lonlims[0] or lon >= lonlims[1]:
            #            lonlims, latlims, tlims = gem.interpfield_lnd(lon, lat, z, t)
            print('WARNING: satGEM field functions must be regenerated')
            break

        if lat <= latlims[0] or lat >= latlims[1]:
            #            lonlims, latlims, tlims = gem.interpfield_lnd(lon, lat, z, t)
            print('WARNING: satGEM field functions must be regenerated')
            break

        if tnum <= tlims[0] or lat >= tlims[1]:
            #            lonlims, latlims, tlims = gem.interpfield_lnd(lon, lat, z, t)
            print('WARNING: satGEM field functions must be regenerated')
            break
        # print(t)

    # After ray tracing loop
    elapsed_time = np.vstack([(timeIn - time[0]).total_seconds()
                              for timeIn in time[:i + 2]])

    # Extend Bathymetry data so you can see whats just beyond end of ray trace
    ext = np.arange(0, extend_bathy, 10)
    x_extend = np.full_like(ext, np.nan)
    y_extend = np.full_like(ext, np.nan)
    b_extend = np.full_like(ext, np.nan)
    for i in range(len(ext)):
        x = x + dx
        y = y + dy
        lon2, lat2 = inverse_hav(dx, dy, lon, lat)
        lon = np.asscalar(lon2)
        lat = np.asscalar(lat2)
        bottom = -gem.bathyF(lon, lat)
        x_extend[i] = x
        y_extend[i] = y
        b_extend[i] = bottom

    # store all results in dictionary (keeps things concise when using)

    results = {
        'x': np.vstack(x_all),
        'y': np.vstack(y_all),
        'z': np.vstack(z_all),
        'k': np.vstack(k_all),
        'l': np.vstack(l_all),
        'm': np.vstack(m_all),
        'omega': np.vstack(om_all),
        'lon': np.vstack(lon_all),
        'lat': np.vstack(lat_all),
        'time': time[:i + 1],
        'elapsed_time': elapsed_time,
        'u': np.vstack(u_all)

    }

    distance = 1e-3 * np.sqrt(results['x']**2 + results['y']**2)
    results['distance'] = distance

    # Had to add this condition for when the run quits out on first step
    if bathy:
        results['bathy'] = np.vstack(bathy)
        results['bathy_ext'] = np.hstack(
            (results['bathy'].flatten(), b_extend))
        results['x_ext'] = np.hstack((results['x'].flatten(), x_extend))
        results['y_ext'] = np.hstack((results['y'].flatten(), y_extend))
        results['cgx'] = np.vstack(cgx)
        results['cgy'] = np.vstack(cgy)
        results['cgz'] = np.vstack(cgz)
        results['N2'] = np.vstack(N2_all)
        results['rx'] = np.vstack(rx_all)
        results['ry'] = np.vstack(ry_all)
        results['rz'] = np.vstack(rz_all)
        results['dudz'] = np.vstack(dudz_all)
        results['dvdz'] = np.vstack(dvdz_all)
        if energy:
            results['action'] = action
            results['m_flux'] = action * \
                (np.diff(results['omega'], axis=0) / tstep)

    if N2subgrid is not None:
        results['N2sub'] = N2subgrid
        results['lonvec'] = lonvec
        results['latvec'] = latvec

    summary = """
Ray Tracing Summary
-------------------
Duration: {} hours
Time Step: {} seconds
Distance Traveled: {} km
Time Direction: {} 
""".format(
        duration,
        tstep,
        distance[-1],
        time_direction
    ).strip('[]')

    results['summary'] = summary

    # Store results onto wave object (just to see if this works for now)
    return results
