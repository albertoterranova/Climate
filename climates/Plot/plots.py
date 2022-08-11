from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np


def plot_base_map(score_map):
    low_corner_lat = score_map.latitude.values.min()
    up_corner_lat = score_map.latitude.values.max()
    low_corner_lon = score_map.longitude.values.min()
    up_corner_lon = score_map.longitude.values.max()
    map = Basemap(projection='merc',
                  lon_0=0,
                  lat_0=0,
                  llcrnrlat=low_corner_lat - 1,
                  urcrnrlat=up_corner_lat + 1,
                  llcrnrlon=low_corner_lon - 1,
                  urcrnrlon=up_corner_lon + 1,
                  resolution='l')
    L, l = score_map.longitude, score_map.latitude
    x, y = map(*np.meshgrid(L, l))
    map.drawcoastlines()
    map.drawmapboundary(fill_color='white')
    map.drawcountries(linewidth=1, linestyle='solid', color='k')
    p1 = map.contourf(x=x, y=y, data=score_map.values /
                      np.sqrt(np.nansum(np.square(score_map.values))))

    plt.show()


def plot_2_vs_2(score_map, score_map1):
    fig, axes = plt.subplots(1, 2)
    low_corner_lat = score_map.latitude.values.min()
    up_corner_lat = score_map.latitude.values.max()
    low_corner_lon = score_map.longitude.values.min()
    up_corner_lon = score_map.longitude.values.max()
    map = Basemap(projection='merc',
                  lon_0=0,
                  lat_0=0,
                  llcrnrlat=low_corner_lat - 1,
                  urcrnrlat=up_corner_lat + 1,
                  llcrnrlon=low_corner_lon - 1,
                  urcrnrlon=up_corner_lon + 1,
                  resolution='c', ax=axes[0])
    L, l = score_map.longitude, score_map.latitude
    x, y = map(*np.meshgrid(L, l))
    map.drawcoastlines()
    map.drawcountries(linewidth=1, linestyle='solid', color='k')
    p = map.contourf(x=x, y=y, data=score_map.values)

    map = Basemap(projection='merc',
                  lon_0=0,
                  lat_0=0,
                  llcrnrlat=low_corner_lat - 1,
                  urcrnrlat=up_corner_lat + 1,
                  llcrnrlon=low_corner_lon - 1,
                  urcrnrlon=up_corner_lon + 1,
                  resolution='c', ax=axes[1])
    L, l = score_map.longitude, score_map.latitude
    x, y = map(*np.meshgrid(L, l))
    map.drawcoastlines()
    map.drawcountries(linewidth=1, linestyle='solid', color='k')
    p1 = map.contourf(x=x, y=y, data=score_map1.values)
    plt.show()


def plot_3_vs_3(score_map, score_map1, score_map2):
    fig, axes = plt.subplots(1, 3)
    low_corner_lat = score_map.latitude.values.min()
    up_corner_lat = score_map.latitude.values.max()
    low_corner_lon = score_map.longitude.values.min()
    up_corner_lon = score_map.longitude.values.max()
    map = Basemap(projection='merc',
                  lon_0=0,
                  lat_0=0,
                  llcrnrlat=low_corner_lat - 1,
                  urcrnrlat=up_corner_lat + 1,
                  llcrnrlon=low_corner_lon - 1,
                  urcrnrlon=up_corner_lon + 1,
                  resolution='c', ax=axes[0])
    L, l = score_map.longitude, score_map.latitude
    x, y = map(*np.meshgrid(L, l))
    map.drawcoastlines()
    map.drawcountries(linewidth=1, linestyle='solid', color='k')
    p = map.contourf(x=x, y=y, data=score_map.values)

    map = Basemap(projection='merc',
                  lon_0=0,
                  lat_0=0,
                  llcrnrlat=low_corner_lat - 1,
                  urcrnrlat=up_corner_lat + 1,
                  llcrnrlon=low_corner_lon - 1,
                  urcrnrlon=up_corner_lon + 1,
                  resolution='c', ax=axes[1])
    L, l = score_map.longitude, score_map.latitude
    x, y = map(*np.meshgrid(L, l))
    map.drawcoastlines()
    map.drawcountries(linewidth=1, linestyle='solid', color='k')
    p1 = map.contourf(x=x, y=y, data=score_map1.values)

    map = Basemap(projection='merc',
                  lon_0=0,
                  lat_0=0,
                  llcrnrlat=low_corner_lat - 1,
                  urcrnrlat=up_corner_lat + 1,
                  llcrnrlon=low_corner_lon - 1,
                  urcrnrlon=up_corner_lon + 1,
                  resolution='c', ax=axes[2])
    L, l = score_map.longitude, score_map.latitude
    x, y = map(*np.meshgrid(L, l))
    map.drawcoastlines()
    map.drawcountries(linewidth=1, linestyle='solid', color='k')
    p = map.contourf(x=x, y=y, data=score_map2.values)
    plt.show()
