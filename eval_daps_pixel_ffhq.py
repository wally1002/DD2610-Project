import os
import argparse
import csv
import gc
import yaml
import numpy as np
from PIL import Image
from functools import partial
import scipy.linalg  # Required for stable FID calculation
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

# Metrics
import piq

# OmegaConf for new config style
from omegaconf import OmegaConf


from model import get_model
from forward_operator import get_operator
from main.scheduler import EDMScheduler
from main.pfode import PFODE
from daps_sampler import DAPS

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Paths to the specific files mentioned in your snippet
MODEL_CONFIG_PATH = 'configs/model/ffhq256ddpm.yaml'
SAMPLER_CONFIG_PATH = 'configs/sampler/edm_daps.yaml'

# List of TASK configs to iterate over.
TASK_CONFIGS_LIST = [
    'configs/task/motion_blur.yaml',
    'configs/task/inpainting_rand.yaml',
    'configs/task/nonlinear_blur.yaml',
    'configs/task/phase_retrieval.yaml',
    'configs/task/super_resolution.yaml',
    'configs/task/gaussian_blur.yaml'
]

# Number of stochastic samples per image per task
NUM_SAMPLES = 1 

# ------------------------------------------------------------------------------
# HELPER CLASSES & FUNCTIONS
# ------------------------------------------------------------------------------

class CustomPathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_size, transform=None):
        self.image_paths = image_paths
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        # Resize to match model expectation (usually 256 for FFHQ)
        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)

class InceptionFeatureExtractor(nn.Module):
    """ Extracts features from InceptionV3 for FID calculation. """
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import Inception_V3_Weights
            weights = Inception_V3_Weights.IMAGENET1K_V1
            inception = models.inception_v3(weights=weights)
        except (ImportError, AttributeError):
            inception = models.inception_v3(pretrained=True)

        self.blocks = nn.ModuleList([
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            inception.maxpool1, inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2, inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b,
            inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            inception.avgpool
        ])
        self.eval()

    def forward(self, x):
        # Resize to 299x299 for Inception
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize to [-1, 1] range which Inception typically expects if input is [0,1]
        x = (x - 0.5) / 0.5
        for block in self.blocks:
            x = block(x)
        return x.view(x.size(0), -1)

def compute_fid_stats(feature_extractor, images_tensor, device, batch_size=20):
    """ Computes mu and sigma for FID. Returns numpy arrays. """
    all_features = []
    n_samples = images_tensor.shape[0]
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = images_tensor[i : i + batch_size].to(device)
            features = feature_extractor(batch)
            all_features.append(features.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
            raise ValueError('Imaginary component found in FID calculation')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def tensor_to_np_img(tensor):
    """Converts [-1, 1] tensor to [0, 255] numpy image for saving."""
    img = tensor.clone().detach().cpu()
    img = (img + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img

def get_logger(name="benchmark"):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='eval_daps_pixel_ffhq')
    args = parser.parse_args()
    
    # ==========================================
    # INPUT: YOUR IMAGE PATHS LIST
    # ==========================================
    IMAGE_PATHS = [
         'dataset/test-ffhq/49799.png',
         'dataset/test-ffhq/49503.png',
         'dataset/test-ffhq/49292.png',
         'dataset/test-ffhq/49573.png',
         'dataset/test-ffhq/49302.png',
         'dataset/test-ffhq/49732.png',
         'dataset/test-ffhq/49088.png',
         'dataset/test-ffhq/49684.png',
         'dataset/test-ffhq/49425.png',
         'dataset/test-ffhq/49582.png',
         'dataset/test-ffhq/49997.png',
         'dataset/test-ffhq/49850.png',
         'dataset/test-ffhq/49120.png',
         'dataset/test-ffhq/49767.png',
         'dataset/test-ffhq/49520.png',
         'dataset/test-ffhq/49405.png',
         'dataset/test-ffhq/49893.png',
         'dataset/test-ffhq/49164.png',
         'dataset/test-ffhq/49673.png',
         'dataset/test-ffhq/49945.png'
    ]

    logger = get_logger()
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    logger.info(f"Device set to {device_str}.")

    # -----------------------
    # 1. LOAD MODEL (Once)
    # -----------------------
    logger.info(f"Loading Model from {MODEL_CONFIG_PATH}")
    model_cfg = OmegaConf.load(MODEL_CONFIG_PATH)
    model = get_model(**model_cfg).to(device)
    model.eval()

    # -----------------------
    # 2. LOAD SAMPLER CONFIG
    # -----------------------
    logger.info(f"Loading Sampler Config from {SAMPLER_CONFIG_PATH}")
    sampler_cfg = OmegaConf.load(SAMPLER_CONFIG_PATH)
    
    # -----------------------
    # 3. PREPARE LOGGING
    # -----------------------
    os.makedirs(args.save_dir, exist_ok=True)
    detailed_csv_path = os.path.join(args.save_dir, 'metrics_detailed_daps.csv')
    summary_csv_path = os.path.join(args.save_dir, 'metrics_summary_daps.csv')

    # Initialize CSV headers
    if not os.path.exists(detailed_csv_path):
        with open(detailed_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Image', 'Sample_Idx', 'PSNR', 'SSIM', 'LPIPS'])

    if not os.path.exists(summary_csv_path):
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Avg_PSNR', 'Avg_SSIM', 'Avg_LPIPS', 'FID'])

    # Metrics calculators
    lpips_metric = piq.LPIPS(replace_pooling=True, reduction='none').to(device)
    inception_fe = InceptionFeatureExtractor().to(device)

    # Dataset transformation ([-1, 1])
    # Note: Adjust image_size based on your model config if not 256
    image_size = model_cfg.get('image_size', 256) 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomPathDataset(IMAGE_PATHS, image_size=image_size, transform=transform)
    # Use batch_size=1 for controlled evaluation loop
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # --------------------------------------------------------------------------
    # OUTER LOOP: TASKS
    # --------------------------------------------------------------------------
    for task_config_path in TASK_CONFIGS_LIST:
        logger.info(f"--- Starting Task: {task_config_path} ---")
        
        if not os.path.exists(task_config_path):
            logger.warning(f"Config file not found: {task_config_path}, skipping.")
            continue
            
        task_cfg = OmegaConf.load(task_config_path)
        
        # Determine operator parameters from task config
        pixel_cfg = task_cfg.pixel
        operator_cfg = pixel_cfg.operator
        mcmc_cfg = pixel_cfg.mcmc_sampler_config
        
        operator_name = operator_cfg.name
        
        # Prepare output dirs
        task_out_path = os.path.join(args.save_dir, operator_name)
        os.makedirs(os.path.join(task_out_path, 'input'), exist_ok=True)
        os.makedirs(os.path.join(task_out_path, 'recon'), exist_ok=True)
        os.makedirs(os.path.join(task_out_path, 'label'), exist_ok=True)

        # 4. Initialize Operator
        operator = get_operator(**operator_cfg)
        # operator = operator.to(device)

        # 5. Initialize DAPS Sampler Components
        num_steps = 5 
        edm_scheduler = EDMScheduler(num_steps)
        
        # Initialize DAPS
        daps = DAPS(
            sampler_cfg['annealing_scheduler_config'],
            sampler_cfg['diffusion_scheduler_config'],
            mcmc_cfg
        )

        task_psnr = []
        task_ssim = []
        task_lpips = []
        
        fid_refs = []   # Real images
        fid_recons = [] # Generated images

        # ----------------------------------------------------------------------
        # INNER LOOP: IMAGES
        # ----------------------------------------------------------------------
        for idx, (ref_img, fname_tup) in enumerate(loader):
            fname = fname_tup[0]
            ref_img = ref_img.to(device) # Shape [1, 3, H, W]
            
            # Save Label
            plt.imsave(os.path.join(task_out_path, 'label', fname), tensor_to_np_img(ref_img[0]))

            # 6. Measurement (Forward Operator)
            with torch.no_grad():
                y = operator.measure(ref_img)
             
            # so we only try to save if it has 3 channels and spatial dims, 
            # otherwise skip or save visualization logic specific to task.
            if y.shape == ref_img.shape:
                plt.imsave(os.path.join(task_out_path, 'input', fname), tensor_to_np_img(y[0]))
            
            # Prepare PFODE with current image shape
            # shape tuple needs to be (N, C, H, W)
            current_shape = ref_img.shape 
            pf_ode = PFODE(edm_scheduler, model, current_shape)

            # Store reference for FID ([0,1] range)
            ref_01 = torch.clamp((ref_img + 1.0) / 2.0, 0.0, 1.0)
            
            # ------------------------------------------------------------------
            # SAMPLING LOOP
            # ------------------------------------------------------------------
            # We iterate NUM_SAMPLES times to generate multiple realizations if desired
            for s_i in range(NUM_SAMPLES):
                
                # 7. Generate Initial Noise & Sample
                # Since we are looping batch_size=1, we generate 1 sample.
                x_init = pf_ode.gaussian_prior_x_T(1).to(device)
                
                # with torch.no_grad():
                    # DAPS Sampling
                x_final = daps.daps_sample(model, x_init, operator, y)
                
                # Save Result
                save_name = f"{os.path.splitext(fname)[0]}_s{s_i}.png"
                plt.imsave(os.path.join(task_out_path, 'recon', save_name), tensor_to_np_img(x_final[0]))

                # 8. Metrics
                sample_01 = torch.clamp((x_final + 1.0) / 2.0, 0.0, 1.0)

                # PSNR
                p_val = piq.psnr(sample_01, ref_01, data_range=1.0).item()
                task_psnr.append(p_val)

                # SSIM
                s_val = piq.ssim(sample_01, ref_01, data_range=1.0).item()
                task_ssim.append(s_val)

                # LPIPS
                with torch.no_grad():
                    l_val = lpips_metric(sample_01, ref_01).item()
                task_lpips.append(l_val)

                # Accumulate for FID
                fid_refs.append(ref_01.detach().cpu())
                fid_recons.append(sample_01.detach().cpu())

                # Log to CSV
                with open(detailed_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([operator_name, fname, s_i, p_val, s_val, l_val])

            if (idx + 1) % 5 == 0:
                logger.info(f"Task {operator_name}: Processed {idx + 1}/{len(dataset)} images.")

        # ----------------------------------------------------------------------
        # TASK AGGREGATION & FID
        # ----------------------------------------------------------------------
        logger.info(f"--- Computing FID for {operator_name} ---")
        
        if len(fid_recons) > 0:
            tensor_refs = torch.cat(fid_refs, dim=0)
            tensor_recons = torch.cat(fid_recons, dim=0)

            # Compute stats
            mu_real, sigma_real = compute_fid_stats(inception_fe, tensor_refs, device)
            mu_fake, sigma_fake = compute_fid_stats(inception_fe, tensor_recons, device)
            fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

            avg_p = sum(task_psnr) / len(task_psnr)
            avg_s = sum(task_ssim) / len(task_ssim)
            avg_l = sum(task_lpips) / len(task_lpips)

            logger.info(f"RESULTS: {operator_name} | PSNR: {avg_p:.2f} | SSIM: {avg_s:.4f} | LPIPS: {avg_l:.4f} | FID: {fid_score:.2f}")

            with open(summary_csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([operator_name, avg_p, avg_s, avg_l, fid_score])
        else:
            logger.warning(f"No samples generated for {operator_name}.")

        # Cleanup per task
        del operator, daps, edm_scheduler, pf_ode
        del fid_refs, fid_recons
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("All tasks completed.")

if __name__ == '__main__':
    main()