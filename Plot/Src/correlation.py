import torch

import numpy as np

def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two arrays.
    
    Parameters:
    a (array-like): First array
    b (array-like): Second array
    
    Returns:
    float: Cosine similarity value between -1 and 1
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Handle zero vectors
    return dot_product / (norm_a * norm_b)

def pearson_r_correlation(a, b):
    """
    Compute the Pearson R correlation coefficient between two arrays.
    
    Parameters:
    a (array-like): First array
    b (array-like): Second array
    
    Returns:
    float: Pearson correlation coefficient between -1 and 1
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    numerator = np.sum((a - mean_a) * (b - mean_b))
    denominator = np.sqrt(np.sum((a - mean_a)**2) * np.sum((b - mean_b)**2))
    
    if denominator == 0:
        return 0.0  # Handle case where standard deviation is 0
    return numerator / denominator


def psnr(a, b, data_range=None):
    """
    Compute Peak Signal-to-Noise Ratio between two arrays.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    if data_range is None:
        max_val = max(a.max(), b.max())
        min_val = min(a.min(), b.min())
        data_range = max_val - min_val
        if data_range <= 0:
            data_range = 1.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def structural_similarity(a, b, data_range=None, k1=0.01, k2=0.03):
    """
    Compute a global Structural Similarity Index Measure (SSIM) for two arrays.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if data_range is None:
        max_val = max(a.max(), b.max())
        min_val = min(a.min(), b.min())
        data_range = max_val - min_val
        if data_range <= 0:
            data_range = 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a_sq = np.mean((a - mu_a) ** 2)
    sigma_b_sq = np.mean((b - mu_b) ** 2)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))
    return ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a_sq + sigma_b_sq + c2))

pt_path = '/N/slate/tnn3/DucHGA/tcg-formation/Data/Ibtracs/FIRST_MERRA2_IBTRACS_node8x8.pt'
pt_list = [
    '/N/scratch/tnn3/TMHOA/GCN_nobias_t2/with_cnn_init_v2/A_3_sim/logs/with_cnn_init_v2_A_3_sim/crop8x8_23_A3_with_cnn_20260329_105209/predict_best.pt',
    '/N/scratch/tnn3/TMHOA/GCN_nobias_t2/with_cnn_init_v2/A_3_sim/logs/with_cnn_init_v2_A_3_sim/crop8x8_24_A3_with_cnn_20260329_043659/predict_best.pt',
    '/N/scratch/tnn3/TMHOA/GCN_nobias_t2/with_cnn_init_v2/A_3_sim/logs/with_cnn_init_v2_A_3_sim/crop8x8_42_A3_with_cnn_20260329_074550/predict_best.pt',
]

if 'full' in pt_list[0]:
    time = 10271
else:
    time = 147

arr_list = [torch.load(pt) for pt in pt_list]
arr = torch.stack(arr_list, dim=0).mean(dim=0)
arr = arr/time

# pt_file = '/N/scratch/tnn3/TMHOA/GCN_nobias_t2/with_cnn_init_v2/A_3_sim/logs/with_cnn_init_v2_A_3_sim/crop8x8_42_A3_with_cnn_20260329_074550/predict_last.pt'
# arr = torch.load(pt_file)
# print(arr.shape)

arr = arr.squeeze()
if arr.shape == (12,):
    arr_shape = (3, 4)
    node_size = 20

elif arr.shape == (30,):
    arr_shape = (5, 6)
    node_size = 12

elif arr.shape == (70,):
    arr_shape = (7, 10)
    node_size = 8

arr = arr.reshape(arr_shape).numpy()
arr_flat = arr.reshape(-1)

ibtracs = torch.load(pt_path)
ibtracs = ibtracs.numpy()
ibtracs = ibtracs / ibtracs.sum()
ibtracs = ibtracs[:, ::-1]
ibtracs_flat = ibtracs.reshape(-1)

similarity = cosine_similarity(arr_flat, ibtracs_flat)
print(f"Cosi: {similarity:.2f}")

pearson_r = pearson_r_correlation(arr_flat, ibtracs_flat)
print(f"Pear: {pearson_r:.2f}")

psnr_value = psnr(arr_flat, ibtracs_flat)
print(f"PSNR: {psnr_value:.2f}")

ssim_value = structural_similarity(arr_flat, ibtracs_flat)
print(f"SSIM: {ssim_value:.2f}")