import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors
import torch
import os

# Parameters
data_path = '/N/scratch/tnn3/DATA/ibtracs/FIRST_MERRA2_IBTRACS.csv'  # Dummy path, replace with actual
output_dir = '/N/slate/tnn3/DucHGA/TC-formation/OutTemp/map'  # Dummy path, replace with actual
lon_min, lon_max, lon_step = 100, 150, 5
lat_min, lat_max, lat_step = 0, 30, 4
years = list(range(2017, 2024))

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
df = df[df['ISO_TIME'].dt.year.isin(years)]

# Define grid
lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)
lat_bins = np.arange(lat_min, lat_max + lat_step, lat_step)

# Calculate density
density = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
total_records = len(df)

for i in range(len(lat_bins)-1):
    for j in range(len(lon_bins)-1):
        mask = (df['LAT'] >= lat_bins[i]) & (df['LAT'] < lat_bins[i+1]) & \
               (df['LON'] >= lon_bins[j]) & (df['LON'] < lon_bins[j+1])
        count = mask.sum()
        density[i, j] = count / total_records if total_records > 0 else 0

# Save density array
torch.save(torch.tensor(density), os.path.join(output_dir, 'ibtracs_density.pt'))

# Prepare arr as in the notebook
arr = density
arr_shape = (7, 10)
node_size = 8

arr = arr.reshape(arr_shape).numpy()  # already is

# 1. Calculate the step sizes based on your existing logic
lon_step = node_size * 0.625
lat_step = node_size * 0.5

# 2. Generate the grid edges directly derived from the data's exact shape
# We use arr_shape[1] + 1 and arr_shape[0] + 1 to define the cell boundaries
lon_edges = lon_max - np.arange(arr_shape[1] + 1) * lon_step
lat_edges = lat_max - np.arange(arr_shape[0] + 1) * lat_step

lon_arr, lat_arr = np.meshgrid(lon_edges, lat_edges)

map_ax = Basemap(projection='cyl',
                    llcrnrlat=lat_min, urcrnrlat=lat_max,
                    llcrnrlon=lon_min, urcrnrlon=lon_max,
                    resolution='c')
map_ax.drawcountries(linewidth=0.8)
map_ax.drawcoastlines(linewidth=0.8)
map_lon_grid = np.arange(lon_max, lon_min - 1, - node_size * 0.625)
map_lat_grid = np.arange(lat_max, lat_min - 1, - node_size * 0.5)
map_ax.drawparallels(map_lat_grid, labels=[1,0,0,0],linewidth=0.5)
map_ax.drawmeridians(map_lon_grid, labels=[0,0,0,1],linewidth=0.5)


base_cmap = plt.get_cmap('twilight')

# Sample the first 0.6 of the colormap
first_half_colors = base_cmap(np.linspace(0.0, 0.6, 128))

# Create the new truncated colormap
cmap = mcolors.ListedColormap(first_half_colors)

cs = map_ax.pcolormesh(lon_arr, lat_arr, arr, shading='auto', cmap=cmap, vmin=0, vmax=0.1)

plt.colorbar(cs, orientation='vertical', shrink=0.8, pad=0.05)
map_ax.fillcontinents(color='lightgray', lake_color=None, alpha=1.0)

plt.title('IBTrACS TCG Density (2017-2023)')

# Custom ticks
plt.xticks(map_lon_grid, [f'{int(lon)}°' for lon in map_lon_grid], rotation=45, ha='right')
plt.yticks(map_lat_grid, [f'{int(lat)}°' for lat in map_lat_grid])

plt.savefig(os.path.join(output_dir, 'ibtracs_density.png'), dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot
fig2, ax2 = plt.subplots(figsize=(10, 8))
map_ax2 = Basemap(projection='cyl',
                  llcrnrlat=lat_min, urcrnrlat=lat_max,
                  llcrnrlon=lon_min, urcrnrlon=lon_max,
                  resolution='c', ax=ax2)
map_ax2.drawcountries(linewidth=0.8)
map_ax2.drawcoastlines(linewidth=0.8)

map_ax2.drawparallels(map_lat_grid, labels=[1,0,0,0], linewidth=0.5)
map_ax2.drawmeridians(map_lon_grid, labels=[0,0,0,1], linewidth=0.5)

map_ax2.fillcontinents(color='lightgray', lake_color=None, alpha=1.0)

# Assign colors based on grid density
colors = []
for _, row in df.iterrows():
    lat, lon = row['LAT'], row['LON']
    i = np.digitize(lat, lat_bins) - 1
    j = np.digitize(lon, lon_bins) - 1
    if 0 <= i < density.shape[0] and 0 <= j < density.shape[1]:
        colors.append(density[i, j])
    else:
        colors.append(0)

sc = map_ax2.scatter(df['LON'], df['LAT'], latlon=True, c=colors, cmap=cmap, s=1, vmin=0, vmax=0.1)
plt.colorbar(sc, orientation='vertical', shrink=0.8, pad=0.05)

plt.title('IBTrACS TCG Locations Colored by Density (2017-2023)')

plt.xticks(map_lon_grid, [f'{int(lon)}°' for lon in map_lon_grid], rotation=45, ha='right')
plt.yticks(map_lat_grid, [f'{int(lat)}°' for lat in map_lat_grid])

plt.savefig(os.path.join(output_dir, 'ibtracs_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Plots and density array saved successfully.")