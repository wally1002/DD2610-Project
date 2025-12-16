import os
import argparse
import csv
import gc
import yaml
import numpy as np
from PIL import Image
from functools import partial
import scipy.linalg  
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

# Hugging Face Datasets
from datasets import load_dataset

# Metrics
import piq

# OmegaConf 
from omegaconf import OmegaConf

from model import get_model
from diffusers import UNet2DModel, DDPMScheduler
from model import DiffusionModel, register_model
from forward_operator import get_operator
from main.scheduler import EDMScheduler
from main.pfode import PFODE
from daps_sampler import DAPS

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

MODEL_CONFIG_PATH = 'configs/model/ffhq256ddpm.yaml'
SAMPLER_CONFIG_PATH = 'configs/sampler/edm_daps.yaml'

TASK_CONFIGS_LIST = [
    # 'configs/task/motion_blur.yaml',
    'configs/task/inpainting_rand.yaml',
    'configs/task/nonlinear_blur.yaml',
    'configs/task/phase_retrieval.yaml',
    'configs/task/super_resolution.yaml',
    'configs/task/gaussian_blur.yaml',
    'configs/task/inpainting.yaml',

]

# Number of stochastic samples per image per task
NUM_SAMPLES = 1 

# ------------------------------------------------------------------------------
# HELPER CLASSES & FUNCTIONS
# ------------------------------------------------------------------------------

class HFSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, split="val", num_images=20, image_size=256, transform=None, seed=42):
        """
        Loads a specific subset of images from Hugging Face deterministically.
        
        Args:
            seed (int): The magic number. As long as this stays 42 (or any fixed int), 
                        you will get the exact same images every time.
        """
        self.image_size = image_size
        self.transform = transform
        
        print(f"Loading {num_images} images from Hugging Face (korexyz/celeba-hq-256x256)...")
        try:
            # 1. Load the dataset
            ds = load_dataset("korexyz/celeba-hq-256x256", split=split)
            
            # 2. Shuffle deterministically using the seed
            # This ensures we don't just get the first 20 (which might be similar),
            # but we get the SAME random 20 every time.
            ds_shuffled = ds.shuffle(seed=seed)
            
            # 3. Select the first N from the shuffled list
            max_len = min(len(ds_shuffled), num_images)
            self.dataset = ds_shuffled.select(range(max_len))
            
            print(f"Successfully loaded {len(self.dataset)} images (Deterministically shuffled with seed {seed}).")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        if self.transform:
            img = self.transform(img)
            
        # We can use the real ID from the dataset if available, otherwise generated index
        # 'fname' ensures the output filename is consistent.
        fname = f"img_{idx:05d}.png"
        
        return img, fname

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
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5
        for block in self.blocks:
            x = block(x)
        return x.view(x.size(0), -1)

def compute_fid_stats(feature_extractor, images_tensor, device, batch_size=20):
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
    parser.add_argument('--save_dir', type=str, default='eval_daps_pixel_celebhq')
    args = parser.parse_args()
    
    logger = get_logger()
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    logger.info(f"Device set to {device_str}.")

    # -----------------------
    # 1. LOAD MODEL
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

    # Dataset transformation
    # image_size = model_cfg.get('image_size', 256) 
    image_size = 256
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # -----------------------
    # 4. DATASET (MODIFIED)
    # -----------------------
    # We load 20 images from HuggingFace instead of local paths
    dataset = HFSubsetDataset(
        split="validation", 
        num_images=20, 
        image_size=image_size, 
        transform=transform,
        seed=42 
    )
    
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
        pixel_cfg = task_cfg.pixel
        operator_cfg = pixel_cfg.operator
        mcmc_cfg = pixel_cfg.mcmc_sampler_config
        
        operator_name = operator_cfg.name
        
        if task_config_path == 'configs/task/inpainting.yaml':
            operator_name = "inpainting_box"
        
        task_out_path = os.path.join(args.save_dir, operator_name)
        os.makedirs(os.path.join(task_out_path, 'input'), exist_ok=True)
        os.makedirs(os.path.join(task_out_path, 'recon'), exist_ok=True)
        os.makedirs(os.path.join(task_out_path, 'label'), exist_ok=True)

        operator = get_operator(**operator_cfg)
        # operator = operator.to(device)

        num_steps = 5 
        edm_scheduler = EDMScheduler(num_steps)
        daps = DAPS(
            sampler_cfg['annealing_scheduler_config'],
            sampler_cfg['diffusion_scheduler_config'],
            mcmc_cfg
        )

        task_psnr = []
        task_ssim = []
        task_lpips = []
        
        fid_refs = []
        fid_recons = []

        # ----------------------------------------------------------------------
        # INNER LOOP: IMAGES
        # ----------------------------------------------------------------------
        for idx, (ref_img, fname_tup) in enumerate(loader):
            fname = fname_tup[0]
            ref_img = ref_img.to(device) # [1, 3, H, W]
            
            # Save Label
            plt.imsave(os.path.join(task_out_path, 'label', fname), tensor_to_np_img(ref_img[0]))

            # Measurement
            with torch.no_grad():
                y = operator.measure(ref_img)
             
            if y.shape == ref_img.shape:
                plt.imsave(os.path.join(task_out_path, 'input', fname), tensor_to_np_img(y[0]))
            
            current_shape = ref_img.shape 
            pf_ode = PFODE(edm_scheduler, model, current_shape)

            ref_01 = torch.clamp((ref_img + 1.0) / 2.0, 0.0, 1.0)
            
            # STOCHASTIC SAMPLING LOOP
            for s_i in range(NUM_SAMPLES):
                x_init = pf_ode.gaussian_prior_x_T(1).to(device)
                
                # DAPS Sample
                x_final = daps.daps_sample(model, x_init, operator, y)
                
                # Save Result
                save_name = f"{os.path.splitext(fname)[0]}_s{s_i}.png"
                plt.imsave(os.path.join(task_out_path, 'recon', save_name), tensor_to_np_img(x_final[0]))

                # Metrics
                sample_01 = torch.clamp((x_final + 1.0) / 2.0, 0.0, 1.0)

                p_val = piq.psnr(sample_01, ref_01, data_range=1.0).item()
                task_psnr.append(p_val)

                s_val = piq.ssim(sample_01, ref_01, data_range=1.0).item()
                task_ssim.append(s_val)

                with torch.no_grad():
                    l_val = lpips_metric(sample_01, ref_01).item()
                task_lpips.append(l_val)

                fid_refs.append(ref_01.detach().cpu())
                fid_recons.append(sample_01.detach().cpu())

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

        del operator, daps, edm_scheduler, pf_ode
        del fid_refs, fid_recons
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("All tasks completed.")

if __name__ == '__main__':
    main()
