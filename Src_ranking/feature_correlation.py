"""
Feature Correlation Analysis Script

This script computes pairwise correlation matrices for weather variables from NetCDF files.
It processes NCEP-FNL or MERRA-2 datasets efficiently using multiprocessing and optimized data loading.

Key optimizations:
- Auto-detects variables from sample file instead of hardcoded lists
- Reads full arrays once, then slices by pressure levels using pre-computed indices
- Computes vorticity/divergence on full (nlev, lat, lon) arrays instead of per-level loops
- Uses numpy advanced indexing for efficient data extraction
- Eliminates redundant loops and dictionary overhead

Author: [Your Name]
Date: May 8, 2026
"""

import sys
import os

# Add custom module path for Utils
sys.path.insert(0, '/N/slate/tnn3/DucHGA/TC-formation/Src_model')

import numpy as np
import xarray as xr
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
from Utils.New_features import vorticity, divergence, meshgrid
import random

# ============================================
# User Configuration Section
# ============================================
# Modify these paths and settings as needed for your dataset

# Input directory containing NetCDF files
input_dir = '/N/scratch/tnn3/DATA/nasa-merra2/merra2_extend'
# Output directory for correlation matrix and variable names
output_dir = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base'
# File pattern to match (e.g., '*.nc')
file_glob = '*.nc'

# Dataset type: 'ncep' for NCEP-FNL, 'merra' for MERRA-2
dataset_type = 'merra'  # 'merra' or 'ncep'
# Excel file with pre-computed statistics (mean, std) for standardization
stats_file = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra_extend_statistics.xlsx'

# Number of worker processes for parallel processing
num_workers = cpu_count() - 1

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ============================================
# Dataset-Specific Configuration
# ============================================
# Set wind variable names based on dataset type
if dataset_type == 'merra':
    wind_u, wind_v = 'U', 'V'
elif dataset_type == 'ncep':
    wind_u, wind_v = 'ugrdprs', 'vgrdprs'
else:
    raise ValueError(f"Unknown dataset_type: {dataset_type}")

# Derived variables to compute (vorticity and divergence)
ADD_VAR = ["VOR", "DIV"]

# ============================================
# Auto-Detection of Variables and Levels
# ============================================
# Find all NetCDF files in input directory
nc_files = sorted(glob.glob(os.path.join(input_dir, file_glob)))
if not nc_files:
    raise ValueError(f"No files found in {input_dir}")

# Randomly sample 20000 files if there are more than 20000 files
if len(nc_files) > 20000:
    nc_files = random.sample(nc_files, 20000)
    print(f"Randomly selected {len(nc_files)} files for processing")

# Open a sample file to detect structure
with xr.open_dataset(nc_files[0]) as ds:
    # Detect pressure level dimension name
    lev_dim_name = 'lev' if 'lev' in ds.dims else 'isobaricInhPa'
    if lev_dim_name not in ds.dims:
        raise ValueError("No pressure level dimension found")
    
    # Get pressure level values and create mapping
    lev_idx = ds[lev_dim_name].values
    # Filter levels between 100-1000 hPa, sorted descending
    PRESSURE_LEVELS = sorted([int(l) for l in lev_idx if 100 <= l <= 1000], reverse=True)
    # Create mapping from level value to array index
    level_to_index = {int(v): i for i, v in enumerate(lev_idx)}
    
    # Auto-detect variables: exclude coordinate/time dimensions
    SINGLE_VAR = []  # Variables without pressure levels
    PRESS_VAR = []   # Variables with pressure levels
    exclude_dims = {'lat', 'latitude', 'lon', 'longitude', 'time', lev_dim_name}
    
    for var in sorted(ds.data_vars):
        if var in exclude_dims:
            continue
        var_dims = set(ds[var].dims)
        if lev_dim_name in var_dims:
            PRESS_VAR.append(var)  # Pressure-level variables
        else:
            SINGLE_VAR.append(var)  # Single-level variables
    
    print(f"Auto-detected SINGLE_VAR: {SINGLE_VAR}")
    print(f"Auto-detected PRESS_VAR: {PRESS_VAR}")

# ============================================
# Pre-compute Level Indices for Efficient Slicing
# ============================================
# Convert pressure levels to array indices for numpy advanced indexing
indices = [level_to_index[level] for level in PRESSURE_LEVELS]

# ============================================
# Build Ordered Variable List
# ============================================
# Create ordered list of all variables following Merra2_dataset pattern:
# 1. Single-level vars: var0
# 2. Pressure vars: var{level} for each level
# 3. Derived vars: VOR{level}, DIV{level} for each level
LIST_VAR = [f"{var}0" for var in SINGLE_VAR]
LIST_VAR += [f"{var}{level}" for var in PRESS_VAR for level in PRESSURE_LEVELS]
LIST_VAR += [f"{var}{level}" for var in ADD_VAR for level in PRESSURE_LEVELS]

# ============================================
# Load Pre-computed Statistics
# ============================================
# Read statistics Excel file and create lookup dictionary
stats_df = pd.read_excel(stats_file)
stats_df_dict = {row['Variable']: {'mean': row['Mean'], 'std': row['Std']} for _, row in stats_df.iterrows()}

# Create arrays for mean and std, aligned with LIST_VAR order
MEANS_ARR = np.array([stats_df_dict.get(var, {'mean': 0})['mean'] for var in LIST_VAR])[:, None]
STDS_ARR = np.array([stats_df_dict.get(var, {'std': 1})['std'] for var in LIST_VAR])[:, None]
# Avoid division by zero
STDS_ARR[STDS_ARR == 0] = 1e-12

# ============================================
# Print Configuration Summary
# ============================================
print(f"Dataset type: {dataset_type}")
print(f"Wind variables: U={wind_u}, V={wind_v}")
print(f"Pressure levels: {PRESSURE_LEVELS}")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Total files: {len(nc_files)}")
print(f"Num workers: {num_workers}")
print(f"Total variables: {len(LIST_VAR)}")
print(f"Variables preview: {LIST_VAR[:5]}...\n")


# ============================================
# Helper Function: Process Single File
# ============================================
def process_file(file_path):
    """
    Process a single NetCDF file efficiently.
    
    Args:
        file_path (str): Path to the NetCDF file
        
    Returns:
        tuple: (count_xy, sum_xy, sum_i_given_j, sample_count) for correlation accumulation
               Returns None if processing fails
    """
    try:
        # Open dataset with netCDF4 engine for better performance
        with xr.open_dataset(file_path, engine='netcdf4') as ds:
            # Extract coordinate arrays
            lat = ds['lat'].values if 'lat' in ds.coords else ds['latitude'].values
            lon = ds['lon'].values if 'lon' in ds.coords else ds['longitude'].values
            lat_dim, lon_dim = len(lat), len(lon)
            
            # Initialize feature list for stacking
            feature_list = []
            
            # 1. Add single-level variables (no pressure dependence)
            for var in SINGLE_VAR:
                feature_list.append(ds[var].squeeze().data)
            
            # 2. Add pressure-level variables for all levels at once
            # Use pre-computed indices for efficient numpy slicing
            for var in PRESS_VAR:
                # Extract all pressure levels in correct order: (n_levels, lat, lon)
                feature_list.extend(ds[var].squeeze().data[indices, :, :])
            
            # 3. Compute derived variables (vorticity and divergence)
            # Use full arrays for vectorized computation, but only for selected pressure levels
            u_full = ds[wind_u].squeeze().data[indices, :, :]  # Shape: (n_selected_levels, lat, lon)
            v_full = ds[wind_v].squeeze().data[indices, :, :]  # Shape: (n_selected_levels, lat, lon)
            
            # Create coordinate grids for derivative computation
            lat_grid, lon_grid = meshgrid(lat, lon, len(PRESSURE_LEVELS))
            
            # Compute vorticity and divergence on selected pressure levels
            vor_full = vorticity(u_full, v_full, lat_grid, lon_grid)
            div_full = divergence(u_full, v_full, lat_grid, lon_grid)
            
            # Add derived variables to feature list
            feature_list.extend(vor_full)
            feature_list.extend(div_full)
            
            # 4. Stack all features into 3D array: (n_vars, lat, lon)
            features = np.stack(feature_list, axis=0)
            
            # Reshape to 2D for correlation: (n_vars, lat*lon)
            data_2d = features.reshape(features.shape[0], lat_dim * lon_dim).astype(np.float32)
            
            # 5. Standardization using pre-computed statistics
            valid = ~np.isnan(data_2d)  # Mask for valid (non-NaN) values
            data_std = np.zeros_like(data_2d)
            # In-place standardization: (data - mean) / std
            np.subtract(data_2d, MEANS_ARR, out=data_std, where=valid)
            np.divide(data_std, STDS_ARR, out=data_std, where=valid)
            
            # 6. Compute correlation statistics using matrix operations
            valid_int = valid.astype(np.int64)
            
            # Count valid pairs: (n_vars, n_vars) matrix
            count_xy = valid_int @ valid_int.T
            
            # Sum of products: (n_vars, n_vars) matrix
            sum_xy = data_std @ data_std.T
            
            # Sum for conditional means: (n_vars, n_vars) matrix
            sum_i_given_j = data_std @ valid_int.T
            
            # Total number of spatial samples (pixels)
            n_samp = data_2d.shape[1]
            
            return (count_xy, sum_xy, sum_i_given_j, n_samp)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ============================================
# Main Processing Loop
# ============================================
if __name__ == '__main__':
    print(f"Processing all {len(nc_files)} files for the official run...\n")
    
    # Initialize accumulation arrays
    n_vars = len(LIST_VAR)
    counts = np.zeros((n_vars, n_vars), dtype=np.int64)      # Valid pair counts
    sum_xy = np.zeros((n_vars, n_vars), dtype=np.float64)    # Sum of products
    sum_i_given_j = np.zeros((n_vars, n_vars), dtype=np.float64)  # Sum for means
    total_samples = 0  # Total spatial samples processed
    
    # Process files in parallel using multiprocessing pool
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for efficient parallel processing with progress bar
        for result in tqdm(pool.imap_unordered(process_file, nc_files), 
                          total=len(nc_files), desc="Processing"):
            if result is None:
                continue  # Skip failed files
            
            # Accumulate statistics from this file
            c_xy, s_xy, s_i_j, n_samp = result
            counts += c_xy
            sum_xy += s_xy
            sum_i_given_j += s_i_j
            total_samples += n_samp
    
    # Check if any data was processed
    if total_samples == 0:
        print("No data accumulated. Check input files.")
        sys.exit(1)
    
    print(f"\nTotal pixels processed: {total_samples}")
    print("Computing correlation matrix...")
    
    # ============================================
    # Correlation Matrix Computation
    # ============================================
    # Convert counts to float for division
    count_float = counts.astype(np.float64)
    
    # Handle potential division by zero with numpy error state
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute conditional means
        mean_i_j = sum_i_given_j / count_float      # E[X_i | X_j valid]
        mean_j_i = sum_i_given_j.T / count_float    # E[X_j | X_i valid]
        
        # Compute covariance: Cov(X_i, X_j) = E[X_i*X_j] - E[X_i]*E[X_j]
        cov = (sum_xy / count_float) - (mean_i_j * mean_j_i)
    
    # Extract variances from diagonal of covariance matrix
    var = np.diag(cov)
    std = np.sqrt(np.maximum(var, 0.0))  # Standard deviations
    
    # Compute standard deviation products for correlation denominator
    std_product = np.outer(std, std)
    std_product[std_product == 0] = 1e-12  # Avoid division by zero
    
    # Compute Pearson correlation: corr = cov / (std_i * std_j)
    correlation = cov / std_product
    correlation[counts == 0] = 0.0  # Set correlation to 0 where no valid pairs
    correlation = np.clip(correlation, -1.0, 1.0)  # Clip to valid range
    
    # ============================================
    # Save Outputs
    # ============================================
    # Save correlation matrix as numpy array
    corr_path = os.path.join(output_dir, 'correlation_matrix.npy')
    np.save(corr_path, correlation)
    
    # Save variable information and metadata
    var_info_path = os.path.join(output_dir, 'variable_names.txt')
    with open(var_info_path, 'w') as f:
        f.write(f"Total variables: {n_vars}\n")
        f.write(f"Dataset type: {dataset_type}\n")
        f.write(f"Pressure levels: {PRESSURE_LEVELS}\n")
        f.write(f"Total pixels: {total_samples}\n")
        f.write(f"Statistics file: {stats_file}\n\n")
        f.write("Variable indices:\n")
        for idx, name in enumerate(LIST_VAR):
            f.write(f"{idx:03d}: {name}\n")
    
    print(f"Correlation matrix saved to: {corr_path}")
    print(f"Variable names saved to: {var_info_path}")
    print("Processing complete!")