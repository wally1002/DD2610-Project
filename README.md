# DD2610-Project

Deep Learning Advanced Project Replicating [DAPS](https://github.com/zhangbingliang2019/DAPS)

## Installation

```bash
pip install -r requirements.txt
```

## Download Pretrained Models

```bash
bash download_pre_trained_models.sh
```

This downloads all necessary models, FFHQ and ImageNet datasets. For CelebA-HQ, we load it directly from Hugging Face.

## Running Evaluation

### FFHQ Dataset (Pixel Space)
```bash
python eval_daps_pixel_ffhq.py
```

### FFHQ Dataset (Latent Space)
```bash
python eval_daps_latent_ffhq.py
```

### ImageNet Dataset (Pixel Space)
```bash
python eval_daps_pixel_imagenet.py
```
### ImageNet Dataset (Latent Space)
```bash
python eval_daps_latent_imagenet.py
```
### CelebA-HQ Dataset (FFHQ Pretrained Model)
```bash
python eval_daps_pixel_celebhq.py
```
### CelebA-HQ Dataset (celebA-HQ Pretrained Model)
```bash
python eval_daps_pixel_celebhq_ownmodel.py
```

## Available Tasks

All tasks defined in the configuration files can be run by executing the corresponding evaluation scripts. Simply modify the configuration file name in the scripts as indicated in the comments.

## Scribble Inpainting

To run scribble inpainting tasks, just change the task configuration file, and if you want a different scribbling mask, you can do so by modifying the mask image name in the configuration file using different images from the `Irregular_mask` directory.

## Configuration Files

Configuration files are located in the `configs/` directory. To run different tasks and modify parameters:
1. Edit the relevant configuration file
2. Update the configuration file name in the evaluation script (see comments in scripts for exact location)
3. Run the evaluation script

## Notebooks

Detailed explanations of the DAPS sampler usage and step-by-step examples are provided in Jupyter notebooks:
- Notebook covering all steps to run on FFHQ and ImageNet (pixel and latent space) ```test_sampler_ffhq_imagenet.ipynb ```
- Notebook for CelebA-HQ with different pretrained models ```test_sampler_celebahq_OOD.ipynb ```

## Note

For any task, simply change the configuration file name in the script where indicated, and all configurations are well-commented to guide you on what to modify.
