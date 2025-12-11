from functools import partial
import os
import argparse
import yaml
import csv
import gc
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Metrics
import piq

# Guided Diffusion imports
# Ensure your python path includes the guided-diffusion root directory
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# The 6 tasks defined in your prompt
TASK_CONFIGS_LIST = [
    'configs/motion_deblur_config.yaml',
    'configs/inpainting_config.yaml',
    'configs/nonlinear_deblur_config.yaml',
    'configs/phase_retrieval_config.yaml',
    'configs/super_resolution_config.yaml',
    'configs/gaussian_deblur_config.yaml'
]

# Number of stochastic samples per image per task
NUM_SAMPLES = 4 

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
        # Resize to match model expectation
        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)

class InceptionFeatureExtractor(nn.Module):
    """ Extracts features from InceptionV3 for FID calculation. """
    def __init__(self):
        super().__init__()
        # Suppress warnings from torchvision about pretrained weights
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        inception = models.inception_v3(weights=weights)
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
        # Note: Your model output is [-1, 1]. The standard Inception preprocessing is often [0,1] -> Normalize.
        # But here we assume input `x` is already in range [0, 1].
        x = (x - 0.5) / 0.5
        for block in self.blocks:
            x = block(x)
        return x.view(x.size(0), -1)

def compute_fid_stats(feature_extractor, images_tensor, device, batch_size=20):
    """ Computes mu and sigma for FID. Handles batching to avoid OOM. """
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
    
    # Handle scalar sigma case (1 feature) just in case, though unlikely for Inception
    if sigma.ndim == 0: 
        sigma = sigma.reshape(1, 1)
        
    return torch.tensor(mu).to(device), torch.tensor(sigma).to(device)

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True, help="Path to model config")
    parser.add_argument('--diffusion_config', type=str, required=True, help="Path to diffusion config")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results_benchmark')
    args = parser.parse_args()
   
    # ==========================================
    # INPUT: YOUR IMAGE PATHS LIST
    # ==========================================
    IMAGE_PATHS = [
         '../../DD2610-Project/dataset/test-ffhq/49799.png',
         '../../DD2610-Project/dataset/test-ffhq/49503.png',
         '../../DD2610-Project/dataset/test-ffhq/49292.png',
         '../../DD2610-Project/dataset/test-ffhq/49573.png',
         '../../DD2610-Project/dataset/test-ffhq/49302.png',
         '../../DD2610-Project/dataset/test-ffhq/49732.png',
         '../../DD2610-Project/dataset/test-ffhq/49088.png',
         '../../DD2610-Project/dataset/test-ffhq/49684.png',
         '../../DD2610-Project/dataset/test-ffhq/49425.png',
         '../../DD2610-Project/dataset/test-ffhq/49582.png',
         '../../DD2610-Project/dataset/test-ffhq/49997.png',
         '../../DD2610-Project/dataset/test-ffhq/49850.png',
         '../../DD2610-Project/dataset/test-ffhq/49120.png',
         '../../DD2610-Project/dataset/test-ffhq/49767.png',
         '../../DD2610-Project/dataset/test-ffhq/49520.png',
         '../../DD2610-Project/dataset/test-ffhq/49405.png',
         '../../DD2610-Project/dataset/test-ffhq/49893.png',
         '../../DD2610-Project/dataset/test-ffhq/49164.png',
         '../../DD2610-Project/dataset/test-ffhq/49673.png',
         '../../DD2610-Project/dataset/test-ffhq/49945.png'
    ]
    
    # Sanity Check
    if len(IMAGE_PATHS) == 0 or not os.path.exists(IMAGE_PATHS[0]):
        print("WARNING: Please populate IMAGE_PATHS in the script with valid file paths.")
        # For testing purposes, if empty, we might return or crash.
    
    logger = get_logger()
    
    # Device setup
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load shared configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    
    # Prepare CSV logging
    os.makedirs(args.save_dir, exist_ok=True)
    detailed_csv_path = os.path.join(args.save_dir, 'metrics_detailed.csv')
    summary_csv_path = os.path.join(args.save_dir, 'metrics_summary.csv')

    # Initialize CSV headers
    with open(detailed_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Image', 'Sample_Idx', 'PSNR', 'SSIM', 'LPIPS'])

    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Avg_PSNR', 'Avg_SSIM', 'Avg_LPIPS', 'FID'])

    # Initialize Metric Calculators
    # LPIPS needs to remain on device (VGG is heavy, load once)
    lpips_metric = piq.LPIPS(replace_pooling=True, reduction='none').to(device)
    
    # Inception for FID (Load once)
    inception_fe = InceptionFeatureExtractor().to(device)

    # --------------------------------------------------------------------------
    # OUTER LOOP: TASKS
    # --------------------------------------------------------------------------
    for task_config_path in TASK_CONFIGS_LIST:
        logger.info(f"--- Starting Task: {task_config_path} ---")
        
        # Load Task Config
        if not os.path.exists(task_config_path):
            logger.warning(f"Config file not found: {task_config_path}, skipping.")
            continue
            
        task_config = load_yaml(task_config_path)
        measure_config = task_config['measurement']
        operator_name = measure_config['operator']['name']
        
        # Create Output Directories for this task
        task_out_path = os.path.join(args.save_dir, operator_name)
        os.makedirs(task_out_path, exist_ok=True)
        for d in ['input', 'recon', 'label']:
            os.makedirs(os.path.join(task_out_path, d), exist_ok=True)

        # 1. Initialize Model & Diffusion (Re-init to ensure clean state or if config varied)
        # Note: If model_config is static, we could load model once outside, 
        # but operator/conditioning often requires fresh setup.
        model = create_model(**model_config)
        model = model.to(device)
        model.eval()
        
        # 2. Prepare Operator & Noise
        operator = get_operator(device=device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])
        
        # 3. Prepare Conditioning
        cond_config = task_config['conditioning']
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
        base_measurement_cond_fn = cond_method.conditioning
        
        # 4. Sampler
        sampler = create_sampler(**diffusion_config)
        base_sample_fn = partial(sampler.p_sample_loop, model=model)

        # 5. Data Loader
        # Create fresh dataset/loader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CustomPathDataset(IMAGE_PATHS, image_size=model_config['image_size'], transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Inpainting mask generator if needed
        mask_gen = None
        if operator_name == 'inpainting':
            mask_gen = mask_generator(**measure_config['mask_opt'])

        # Metric Storage for this Task
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
            ref_img = ref_img.to(device)
            
            # Prepare Measurement (y) once per image
            if operator_name == 'inpainting':
                mask = mask_gen(ref_img)
                mask = mask[:, 0, :, :].unsqueeze(dim=0)
                # Update conditioning function with mask
                measurement_cond_fn = partial(base_measurement_cond_fn, mask=mask)
                y = operator.forward(ref_img, mask=mask)
            else:
                measurement_cond_fn = base_measurement_cond_fn
                y = operator.forward(ref_img)
            
            y_n = noiser(y) # Noisy measurement

            # Bind conditioning to sample function
            sample_fn = partial(base_sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Save input/label once per image (visual check)
            plt.imsave(os.path.join(task_out_path, 'input', fname), clear_color(y_n))
            plt.imsave(os.path.join(task_out_path, 'label', fname), clear_color(ref_img))

            # Store reference for FID (normalization [0,1] needed for metrics usually)
            # clear_color returns numpy uint8, but for FID tensor calculation we keep tensors
            # Ref range [-1, 1] -> [0, 1]
            ref_01 = torch.clamp((ref_img + 1.0) / 2.0, 0.0, 1.0)
            
            # We add the reference ONCE per image (20 total) to the "Real" pile
            # OR typically we match the number of samples. 
            # If we generate 80 fakes, we should ideally compare against 80 reals (repeats).
            # Here we will append `NUM_SAMPLES` copies of ref to balance the distribution size.
            for _ in range(NUM_SAMPLES):
                fid_refs.append(ref_01.detach().cpu())

            # ------------------------------------------------------------------
            # SAMPLING LOOP (NUM_SAMPLES times)
            # ------------------------------------------------------------------
            for s_i in range(NUM_SAMPLES):
                # Generate random start noise
                x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
                
                # Run Diffusion
                # We disable 'record' to save disk space, only save final
                sample = sample_fn(x_start=x_start, measurement=y_n, record=False)
                
                # Save visual result
                save_name = f"{os.path.splitext(fname)[0]}_s{s_i}.png"
                plt.imsave(os.path.join(task_out_path, 'recon', save_name), clear_color(sample))

                # Prepare for Metrics ([0, 1] range)
                sample_01 = torch.clamp((sample + 1.0) / 2.0, 0.0, 1.0)

                # 1. PSNR
                p_val = piq.psnr(sample_01, ref_01, data_range=1.0).item()
                task_psnr.append(p_val)

                # 2. SSIM
                s_val = piq.ssim(sample_01, ref_01, data_range=1.0).item()
                task_ssim.append(s_val)

                # 3. LPIPS
                with torch.no_grad():
                    l_val = lpips_metric(sample_01, ref_01).item()
                task_lpips.append(l_val)

                # Store for FID
                fid_recons.append(sample_01.detach().cpu())

                # Log Detail
                with open(detailed_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([operator_name, fname, s_i, p_val, s_val, l_val])

            logger.info(f"Task: {operator_name} | Img: {fname} | Completed {NUM_SAMPLES} samples.")

        # ----------------------------------------------------------------------
        # END OF TASK AGGREGATION
        # ----------------------------------------------------------------------
        logger.info(f"--- Computing FID for {operator_name} ---")
        
        # Concatenate lists to tensors
        # Move back to GPU in batches inside 'compute_fid_stats' to save RAM
        tensor_refs = torch.cat(fid_refs, dim=0)   # [20*4, 3, H, W]
        tensor_recons = torch.cat(fid_recons, dim=0) # [20*4, 3, H, W]

        mu_real, sigma_real = compute_fid_stats(inception_fe, tensor_refs, device)
        mu_fake, sigma_fake = compute_fid_stats(inception_fe, tensor_recons, device)

        fid_metric = piq.FID()
        # piq.FID expects statistics
        fid_score = fid_metric(mu_real, sigma_real, mu_fake, sigma_fake).item()

        # Averages
        avg_p = sum(task_psnr) / len(task_psnr)
        avg_s = sum(task_ssim) / len(task_ssim)
        avg_l = sum(task_lpips) / len(task_lpips)

        logger.info(f"TASK RESULTS: {operator_name}")
        logger.info(f"PSNR: {avg_p:.4f} | SSIM: {avg_s:.4f} | LPIPS: {avg_l:.4f} | FID: {fid_score:.4f}")

        # Save Summary
        with open(summary_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([operator_name, avg_p, avg_s, avg_l, fid_score])

        # Cleanup Memory for next task
        del model, operator, noiser, cond_method, sampler
        del tensor_refs, tensor_recons
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("All tasks completed.")

if __name__ == '__main__':
    main()