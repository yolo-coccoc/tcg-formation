import numpy as np

OMEGA = 7.29 * 1e-5

def vorticity(u, v, lat, lon):
    x = lon
    y = lat
    
    y[y == 0] = 0.5
    
    lat = np.deg2rad(lat)
    
    vx = v / x
    uy = u / y
    
    return vx - uy + 2 * OMEGA * np.sin(lat) * 111 * 1000

def divergence(u, v, lat, lon):
    x = lon
    y = lat
    
    y[y == 0] = 0.5
    
    ux = u / x
    vy = v / y
    
    return ux + vy

def meshgrid(lat, lon, lvl=1):
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lat_grid = np.repeat(lat_grid[np.newaxis, :, :], lvl, axis=0)
    lon_grid = np.repeat(lon_grid[np.newaxis, :, :], lvl, axis=0)
    
    return lat_grid, lon_grid