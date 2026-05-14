import os
import pandas as pd
from pathlib import Path

def parse_positive_file(filename):
    """Parse POSITIVE_1980076N06148.nc format"""
    try:
        # Remove .nc extension
        name = filename.replace('.nc', '')
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) >= 2 and parts[0] == 'POSITIVE':
            file_id = parts[1]
            year = file_id[:4]
            return {
                'ID': file_id,
                'Year': year,
                'Domain': 'POSITIVE',
                'Position': 0,
                'Step': 0
            }
    except:
        pass
    return None

def parse_pastdomain_file(filename):
    """Parse NEGATIVE_1980076N06148_10_19800314_1800.nc format"""
    try:
        # Remove .nc extension
        name = filename.replace('.nc', '')
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) >= 4 and parts[0] == 'NEGATIVE':
            file_id = parts[1]
            year = file_id[:4]
            step = parts[2]
            return {
                'ID': file_id,
                'Year': year,
                'Domain': 'PastDomain',
                'Position': 0,
                'Step': step
            }
    except:
        pass
    return None

def parse_dynamicdomain_file(filename):
    """Parse NEGATIVE_1980076N06148_e_14.nc format"""
    try:
        # Remove .nc extension
        name = filename.replace('.nc', '')
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) >= 4 and parts[0] == 'NEGATIVE':
            file_id = parts[1]
            year = file_id[:4]
            direction_char = parts[2]
            step = parts[3]
            
            # Convert direction character to position (1-8)
            direction_map = {'n': 1, 'ne': 2, 'e': 3, 'se': 4, 's': 5, 'sw': 6, 'w': 7, 'nw': 8}
            position = direction_map.get(direction_char, 0)
            
            return {
                'ID': file_id,
                'Year': year,
                'Domain': 'DynamicDomain',
                'Position': position,
                'Step': step
            }
    except:
        pass
    return None

def organize_data(input_folder):
    """Walk through input folder and collect file information"""
    data = []
    
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if not filename.endswith('.nc'):
                continue
            
            full_path = os.path.join(root, filename)
            domain = os.path.basename(root)
            
            file_info = None
            
            if domain == 'POSITIVE':
                file_info = parse_positive_file(filename)
            elif domain == 'PastDomain':
                file_info = parse_pastdomain_file(filename)
            elif domain == 'DynamicDomain':
                file_info = parse_dynamicdomain_file(filename)
            
            if file_info:
                file_info['FullPath'] = full_path
                file_info['FileName'] = filename
                data.append(file_info)
    
    return pd.DataFrame(data)

# Main execution
if __name__ == '__main__':
    input_folder = '/N/scratch/tnn3/DATA/ncep-fnl/ncep_domain'  # Update with your actual path
    
    df = organize_data(input_folder)
    
    print(len(df))
    # Reorder columns
    column_order = ['FullPath', 'FileName', 'ID', 'Year', 'Domain', 'Position', 'Step']
    df = df[column_order]
    df = df.sort_values(by=['Position', 'Step', 'ID']).reset_index(drop=True)
    
    # Export to CSV
    output_path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Ncep/data_path_check.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Data exported to {output_path}")
    print(f"Total files processed: {len(df)}")
    print("\nFirst few rows:")
    print(df.head())