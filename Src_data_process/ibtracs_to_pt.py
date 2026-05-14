import pandas as pd
import numpy as np
import torch
import os

# Parameters
data_path = '/N/scratch/tnn3/DATA/ibtracs/FIRST_MERRA2_IBTRACS.csv'  # Path to IBTrACS CSV file
output_path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Ibtracs/FIRST_MERRA2_IBTRACS_node20x20.pt'  # Output .pt file path
lat_min, lat_max, lat_step = 0, 30, 10  # Latitude range and step (from 0 to 30, step 4)
lon_min, lon_max, lon_step = 100, 150, 12.5  # Longitude range and step (from 100 to 150, step 5)
years = list(range(2017, 2023))  # Years to include (2017 to 2023)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load data
df = pd.read_csv(data_path)
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
df = df[df['ISO_TIME'].dt.year.isin(years)]

# Define grid bins
lat_bins = np.arange(lat_max, lat_min - 1, -lat_step)  # [30, 26, 22, 18, 14, 10, 6, 2]
lon_bins = np.arange(lon_min, lon_max + 1, lon_step)  # [100, 105, ..., 150]

print(f"Latitude bins: {lat_bins}")
print(f"Longitude bins: {lon_bins}")

# Function to determine lat_idx (optimized)
def get_lat_idx(lat):
    if lat > lat_max:
        return -1
    idx = int((lat_max - lat) // lat_step)
    if idx >= len(lat_bins) - 1:
        idx = len(lat_bins) - 1
    return idx

# Determine grid for each record
# df['lat_idx'] = df['LAT'].apply(get_lat_idx)
# df['lon_idx'] = pd.cut(df['LON'], bins=lon_bins, labels=False, right=False)

df['lat_idx'] = ((lat_max - df['LAT']) // lat_step).astype(int)
df['lon_idx'] = ((df['LON'] - lon_min) // lon_step).astype(int)

# Group by grid and count
grouped = df.groupby(['lat_idx', 'lon_idx']).size().reset_index(name='count')

# Transform to 2D array
count_array = np.zeros((len(lat_bins)-1, len(lon_bins)-1), dtype=int)
for _, row in grouped.iterrows():
    i = int(row['lat_idx'])
    j = int(row['lon_idx'])
    count = int(row['count'])
    if 0 <= i < count_array.shape[0] and 0 <= j < count_array.shape[1]:
        count_array[i, j] = count

# Save as .pt
torch.save(torch.tensor(count_array), output_path)

print(f"2D count array saved to {output_path}")
print(f"Array shape: {count_array.shape}")
print(f"Total records: {count_array.sum()}")