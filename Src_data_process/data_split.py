import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def undersample_data(df_labeled, ratio=30, random_state=42):
    """Random undersampling to achieve 1:ratio 1:0"""
    df_labeled = df_labeled.dropna(subset=['Label'])
    true_samples = df_labeled[df_labeled['Label'] == 1]
    false_samples = df_labeled[df_labeled['Label'] == 0]
    
    n_true = len(true_samples)
    n_false_desired = n_true * ratio
    
    if len(false_samples) > n_false_desired:
        false_samples = false_samples.sample(n=int(n_false_desired), random_state=random_state)
    
    return pd.concat([true_samples, false_samples])

def main(data_path, train_val_years, test_years, steps, ratio, base_path):
    # Load the data
    df = pd.read_csv(data_path)
    
    # Convert types
    df['Step'] = df['Step'].astype(int)
    df['Position'] = df['Position'].astype(int)
    df['Year'] = df['Year'].astype(int)
    
    # Initialize Label column
    df['Label'] = np.nan
    
    # Keep original dataframe for maintaining order
    df_original = df.copy()
    
    for step in steps:
        df = df_original.copy()
        # Labeling logic
        # Set Label to 1 for data at Step == step and Position == 0
        mask_1 = (df['Step'] == step) & (df['Position'] == 0)
        df.loc[mask_1, 'Label'] = 1
        
        # Set Label to 0 for data at Step from step+1 to 20 and Position == 0
        mask_0_pos0 = (df['Step'] > step) & (df['Step'] <= 20) & (df['Position'] == 0)
        df.loc[mask_0_pos0, 'Label'] = 0
        
        # Set Label to 0 for data at Step from step to 20 and Position > 0
        mask_0_pos_gt0 = (df['Step'] >= step) & (df['Step'] <= 20) & (df['Position'] > 0)
        df.loc[mask_0_pos_gt0, 'Label'] = 0
        
        # Filter for train/val and test
        train_val_df = df[df['Year'].isin(train_val_years)].copy()
        test_df = df[df['Year'].isin(test_years)].copy()
        
        # Undersample train/val
        train_val_balanced = undersample_data(train_val_df, ratio)
        
        # Split train/val 9:1 on indices
        train_selected_indices, val_selected_indices = train_test_split(train_val_balanced.index, test_size=0.1, random_state=42, stratify=train_val_balanced['Label'])
        
        # Undersample testRus
        test_balanced = undersample_data(test_df, ratio)
        
        # Create full tables maintaining original order from df_original
        train_full = df_original.copy()
        train_full['Label'] = np.nan
        train_full.loc[train_selected_indices, 'Label'] = df.loc[train_selected_indices, 'Label']
        
        val_full = df_original.copy()
        val_full['Label'] = np.nan
        val_full.loc[val_selected_indices, 'Label'] = df.loc[val_selected_indices, 'Label']
        
        test_full = df_original.copy()
        test_full['Label'] = np.nan
        test_full.loc[test_df.index, 'Label'] = test_df['Label']
        
        test_rus = df_original.copy()
        test_rus['Label'] = np.nan
        test_rus.loc[test_balanced.index, 'Label'] = test_balanced['Label']
        
        # Print counts
        print(f"For step {step}:")
        print(f"  Train: 1s: {train_full['Label'].eq(1).sum()}, 0s: {train_full['Label'].eq(0).sum()}, NaNs: {train_full['Label'].isna().sum()}")
        print(f"  Val: 1s: {val_full['Label'].eq(1).sum()}, 0s: {val_full['Label'].eq(0).sum()}, NaNs: {val_full['Label'].isna().sum()}")
        print(f"  TestFull: 1s: {test_full['Label'].eq(1).sum()}, 0s: {test_full['Label'].eq(0).sum()}, NaNs: {test_full['Label'].isna().sum()}")
        print(f"  TestRus: 1s: {test_rus['Label'].eq(1).sum()}, 0s: {test_rus['Label'].eq(0).sum()}, NaNs: {test_rus['Label'].isna().sum()}")
        
        # Create folder for this step
        step_folder = os.path.join(base_path, f'Step_{step}')
        os.makedirs(step_folder, exist_ok=True)
        
        # Export to CSV in the step folder
        train_full.to_csv(os.path.join(step_folder, 'train.csv'), index=False)
        val_full.to_csv(os.path.join(step_folder, 'val.csv'), index=False)
        test_full.to_csv(os.path.join(step_folder, 'testFull.csv'), index=False)
        test_rus.to_csv(os.path.join(step_folder, 'testRus.csv'), index=False)
        
        print(f"Datasets for step {step} created and exported to {step_folder}.\n")

if __name__ == '__main__':
    # Input arguments
    data_path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Ncep/Base/ncep_domain_full_path.csv'
    train_val_years = range(1980, 2017)  # 1980-2014
    test_years = range(2017, 2023)       # 2015-2020
    steps = list(range(0, 11, 1))       # 0, 2
    ratio = 20
    base_path = f'/N/slate/tnn3/DucHGA/TC-formation/Data/Ncep/Dataset/Rus{ratio}/'
    
    main(data_path, train_val_years, test_years, steps, ratio, base_path)