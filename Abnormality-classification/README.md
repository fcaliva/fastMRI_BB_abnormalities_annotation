# Abnormality classification 
This code reproduces the real-world abnormality classification results in our paper **"Adversarial Robust Training of Deep Learning MRI Reconstruction Models"**.
  * **Author**: Francesco Caliva', Kaiyang Victor Cheng, Rutwik Shah and Valentina Pedoia 
  * **Email**: francesco.caliva@ucsf.edu

## Patch Data preparation
Data are processed by means of the dataloader in `src/dataset_recon.py`. 
This internally uses `src/datasets.py`, which links to the actual data-paths through the `get_dataloaders` class. 

The path to the actual data is specified in `get_dataloaders`, using `.pickle` files. 
Each pickle file is a list of arrays, where each array has 2 elements: path/to/patchfile.h5 and binary label (i.e. presence/absence of abnormality in the patch). 

Patches can be extracted from reconstructed images using the code in `src/extract_patches_on_recon_results.py`

---

## Model checkpoints can be downloaded from:
    squeeze net trained using patches 32x32 
    https://ucsf.box.com/s/rzok5z0fx7hcz0rl7qvraujpvhuu9j6p
    squeeze net trained using patches 64x64
    https://ucsf.box.com/s/rzok5z0fx7hcz0rl7qvraujpvhuu9j6p 

**Disclosure**: We think it's important to remark that both models were trained using the fully-sampled from the fastMRI dataset.
As reported in our manuscript in Section 5 Limitation, we observed a noticeable drop in classification performance when images reconstructed with a significantly better reconstruction method, such as I-RIM were fed. This drop in performance can be associated to a covariate shift resulting from different image reconstruction techniques; as part of our future work we will analyze whether domain adaptation techniques would mitigate the drop in performance which was observed.

## Run
Our paper results can be replicated with the following commands:
---
**Using patches of size 32x32**

**Acceleration Factor 4x:**
---
**GT**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'gt' --feature_extractor --experiment_desc "GT"
---
**U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'unet4' --feature_extractor --experiment_desc "unet4x" 
---
**FNAF U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'fnaf4' --feature_extractor --experiment_desc "fnaf_unet4x" 
---
**B-BOX U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'unet4x_on_abnormality_bb' --feature_extractor --experiment_desc "b-box_unet4x" 
---
**I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'irim4' --feature_extractor --experiment_desc "irim4x" 
---
**FNAF I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'fnaf_irim4' --feature_extractor --experiment_desc "fnaf_irim4x" 
---
**B-BOX I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'irim4x_on_abnormality_bb' --feature_extractor --experiment_desc "b-box_irim4x" 
---

**Acceleration Factor 8X:**

**U-NET**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'unet8' --feature_extractor --experiment_desc "unet8x" 
---
**FNAF U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'fnaf8' --feature_extractor --experiment_desc "fnaf_unet8x" 
---
**B-BOX U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'unet8x_on_abnormality_bb' --feature_extractor --experiment_desc "b-box_unet8x" 
---
**I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'irim8' --feature_extractor --experiment_desc "irim8x" 
---
**FNAF I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'fnaf_irim8' --feature_extractor --experiment_desc "fnaf_irim8x" 
---
**B-BOX I-RIM**

  python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_167/checkpoint.pt' --visible_gpus 0 --patch_size 32 --extract_representation --use_net 'squeezenet' --recon_approach 'irim8x_on_abnormality_bb_new' --feature_extractor --experiment_desc "b-box_irim8x" 

---
**Using patches of size 64x64**

**Acceleration Factor 4x:**
---
**GT**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'gt' --feature_extractor --experiment_desc "GT" 
---
**U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'unet4' --feature_extractor --experiment_desc "unet4x" 
---
**FNAF U-Net**

    python main.py --n]o_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'fnaf4' --feature_extractor --experiment_desc "fnaf_unet4x" 

---
**B-BOX U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'unet4x_on_abnormality_bb' --feature_extractor --experiment_desc "bbox_unet4x" 

---
**I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'irim4' --feature_extractor --experiment_desc "irim4x" 

---
**FNAF I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'fnaf_irim4' --feature_extractor --experiment_desc "fnaf_irim4x" 

---
**B-BOX I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'irim4x_on_abnormality_bb' --feature_extractor --experiment_desc "b-box_irim4x" 
---

**Acceleration Factor 8X:**

---
**U-NET**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'unet8' --feature_extractor --experiment_desc "unet8x"

---
**FNAF U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'fnaf8' --feature_extractor --experiment_desc "fnaf_unet8x"

---
**B-BOX U-Net**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'unet8x_on_abnormality_bb' --feature_extractor --experiment_desc "b-box_unet8x" 

---
**I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'irim8' --feature_extractor --experiment_desc "irim8x" 

---
**FNAF I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'fnaf_irim8' --feature_extractor --experiment_desc "fnaf_irim8x"

---    
**B-BOX I-RIM**

    python main.py --no_distributed --inference --load_checkpoint_dir 'experiments/Model_170/checkpoint.pt' --visible_gpus 0 --patch_size 64 --extract_representation --use_net 'squeezenet' --recon_approach 'irim8x_on_abnormality_bb' --feature_extractor --experiment_desc "b-box_irim8x"

## Re-training example
Training can be launched using the following command:

    python main.py --learning_rate 0.001 --decay_rate 0.9 --decay_steps 500 --no_distributed --experiment_desc 'trainining using patchsize 32x32' --use_net 'squeezenet' --visible_gpus 0 --recon_approach 'gt' --patch_size 32

    python main.py --learning_rate 0.001 --decay_rate 0.9 --decay_steps 500 --no_distributed --experiment_desc 'trainining using patchsize 64x64' --use_net 'squeezenet' --visible_gpus 0 --recon_approach 'gt' --patch_size 64


**Credits**: This code was adapted for our purposes, from the Pytorch Boilerplate:\
  https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate\
  * **Author**: Fabio De Sousa Ribeiro
  * **Email**: fdesosuaribeiro@lincoln.ac.uk
