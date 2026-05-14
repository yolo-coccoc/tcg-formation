import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import re
import multiprocessing
from tqdm import tqdm


def process_task(task):
    f, dt, centers, lat, lon, half_h, half_w, out_dir, window_height_pix, window_width_pix = task
    
    ds = xr.open_dataset(f)
    metadata_list = []
    
    for lat_idx, lon_idx in centers:
        # Calculate crop indices
        lat_start = max(0, lat_idx - half_h)
        lat_end = min(len(lat) - 1, lat_idx + half_h) + 1
        lon_start = max(0, lon_idx - half_w)
        lon_end = min(len(lon) - 1, lon_idx + half_w) + 1

        # Crop the dataset
        cropped = ds.isel(latitude=slice(lat_start, lat_end), longitude=slice(lon_start, lon_end))

        # Get center coordinates
        center_lat = lat[lat_idx]
        center_lon = lon[lon_idx]

        # Create center directory
        center_dir = os.path.join(out_dir, f"{center_lat:.3f}_{center_lon:.3f}")
        os.makedirs(center_dir, exist_ok=True)

        # Save cropped data
        out_file = os.path.join(center_dir, f"{dt.strftime('%Y%m%d_%H%M')}.nc")
        cropped.to_netcdf(out_file)

        # Collect metadata
        metadata_list.append({
            'Datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'Point': f"{center_lat:.3f}_{center_lon:.3f}",
            'Path': out_file,
            'Step': 0,
            'Position': 0,
            'Label': 0
        })
    
    ds.close()
    return metadata_list


def main():
    # Set your input parameters here
    num_cpus = 16
    min_lat = 0.0
    max_lat = 30.0
    min_lon = 100.0
    max_lon = 150.0
    step_lat_pix = 10
    step_lon_pix = 8
    window_width_pix = 33
    window_height_pix = 33
    start_date = datetime.strptime('2017-01-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2024-01-01', '%Y-%m-%d').date()
    input_dir = '/N/scratch/tnn3/DATA/nasa-merra2/merra2_extend'
    out_dir = '/N/scratch/tnn3/DATA/nasa-merra2/merra2_slide'
    dataset_type = 'merra'  # set to 'ncep' or 'merra'

    # Find and filter files by datetime
    files = glob.glob(os.path.join(input_dir, '*.nc'))
    files.sort()
    valid_files = []
    for f in files:
        fname = os.path.basename(f)
        if dataset_type == 'ncep':
            match = re.search(r'fnl_(\d{8})_(\d{2})_(\d{2})', fname)
        else:
            match = re.search(r'merra2_(\d{8})_(\d{2})_(\d{2})', fname)

        if match:
            dt_str = f"{match.group(1)}_{match.group(2)}{match.group(3)}"
            dt = datetime.strptime(dt_str, '%Y%m%d_%H%M')
            if start_date <= dt.date() <= end_date:
                valid_files.append((f, dt))

    if not valid_files:
        print("No valid files found in the datetime range.")
        return

    # Load coordinates from the first file
    ds_sample = xr.open_dataset(valid_files[0][0])
    lat = ds_sample['latitude'].values
    lon = ds_sample['longitude'].values
    ds_sample.close()

    # Find indices for the processing area
    lat_mask = (lat >= min_lat) & (lat <= max_lat)
    lon_mask = (lon >= min_lon) & (lon <= max_lon)
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        print("No data points found in the specified area.")
        return

    min_lat_idx = lat_indices.min()
    max_lat_idx = lat_indices.max()
    min_lon_idx = lon_indices.min()
    max_lon_idx = lon_indices.max()

    # Generate center positions
    centers = []
    lat_idx = max_lat_idx  # Start from top (max lat)
    while lat_idx >= min_lat_idx:
        lon_idx = min_lon_idx  # Start from left (min lon)
        while lon_idx <= max_lon_idx:
            centers.append((lat_idx, lon_idx))
            lon_idx += step_lon_pix
        lat_idx -= step_lat_pix  # Move south (towards smaller lat)

    # Calculate window half-sizes
    half_h = (window_height_pix - 1) // 2
    half_w = (window_width_pix - 1) // 2

    # Create date-range output directory
    year_dir = os.path.join(out_dir, f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}")
    os.makedirs(year_dir, exist_ok=True)

    # Prepare tasks
    tasks = []
    for f, dt in valid_files:
        tasks.append((f, dt, centers, lat, lon, half_h, half_w, year_dir, window_height_pix, window_width_pix))

    # Process tasks in parallel
    with multiprocessing.Pool(processes=num_cpus) as pool:
        metadata = []
        for result in tqdm(pool.imap(process_task, tasks), total=len(tasks), desc="Processing files"):
            metadata.extend(result)

    # Save metadata to CSV
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(year_dir, 'metadata.csv')
    df.to_csv(csv_path, index=False)
    print(f"Processing complete. Metadata saved to {csv_path}")


if __name__ == '__main__':
    main()