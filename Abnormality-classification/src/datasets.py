# -*- coding: utf-8 -*-
"""
    Loading of various datasets.
"""
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST

from utils import *
from dataset_recon import fastMRIPatchDataset, Standardize, ToTensor
####################
# %% codecell
import torch
import h5py
import pickle
import pdb

def get_dataloaders(args):
    """ Gets the dataloaders for the chosen dataset.
    """
    if args.dataset== 'patch_anomaly_detection':
        dataset = 'patch_anomaly_detection'
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not args.inference:
            if int(args.patch_size) == 32:
                dataset_paths = {'train': 'fastMRI/patches_stride2_train.pickle',
                                'valid':  'fastMRI/patches_val.pickle',
                                'test':  'fastMRI/patches_val.pickle'}
            elif int(args.patch_size) == 64:
                dataset_paths = {'train':   'fastMRI/patches_size64_stride8_train.pickle',
                                'valid':    'fastMRI/patches_size64_stride8_val.pickle',
                                'test':     'fastMRI/patches_size64_stride8_val.pickle'}            
        elif args.inference:
            if int(args.patch_size) == 32:
                dataset_paths = {'train': 'fastMRI/patches_stride2_train.pickle',
                                 'valid':  'fastMRI/patches_val.pickle',
                                 'test':''}
                if args.recon_approach== 'gt':
                    dataset_paths['test'] =     'fastMRI/patches_val.pickle'
                elif args.recon_approach== 'unet4':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_unet_reconstructions_val_4x.pickle'
                elif args.recon_approach== 'unet8':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_unet_reconstructions_val_8x.pickle'
                elif args.recon_approach== 'fnaf4':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_fn_unet_reconstructions_val_4x.pickle'
                elif args.recon_approach== 'fnaf8':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_fn_unet_reconstructions_val_8x.pickle'
                elif args.recon_approach== 'irim4':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_irim_val_4x_out.pickle'
                elif args.recon_approach== 'irim8':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_irim_val_8x_out.pickle'
                elif args.recon_approach== 'fnaf_irim4':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_irim_fnaf_train_1000pixel_val_4x_out_nmse.pickle'
                elif args.recon_approach== 'fnaf_irim8':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_irim_fnaf_train_1000pixel_val_8x_out_nmse.pickle'
                elif args.recon_approach== 'unet4x_on_abnormality_bb':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_bbox_unet_reconstructions_val_4x.pickle'
                elif args.recon_approach== 'unet8x_on_abnormality_bb':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_bbox_unet_reconstructions_val_8x.pickle'
                elif args.recon_approach== 'irim4x_on_abnormality_bb':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_irim_bbox_train_val_4x_out_nmse.pickle'            
                elif args.recon_approach== 'irim8x_on_abnormality_bb_new':
                    dataset_paths['test'] =     'fastMRI/patches_size32_stride2_irim_bbox_train_val_8x_out_new.pickle'
                else:
                    print('this dataset is not available')
            if int(args.patch_size) == 64:
                dataset_paths = {'train':   'fastMRI/patches_size64_stride8_train.pickle',
                                'valid':    'fastMRI/patches_size64_stride8_val.pickle',
                                'test':         ''}
                if args.recon_approach== 'gt':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_val.pickle'
                elif args.recon_approach== 'unet4':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_unet_reconstructions_val_4x.pickle'
                elif args.recon_approach== 'unet8':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_unet_reconstructions_val_8x.pickle'
                elif args.recon_approach== 'fnaf4':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_fn_unet_reconstructions_val_4x.pickle'
                elif args.recon_approach== 'fnaf8':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_fn_unet_reconstructions_val_8x.pickle'
                elif args.recon_approach== 'irim4':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_irim_val_4x_out.pickle'
                elif args.recon_approach== 'irim8':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_irim_val_8x_out.pickle'
                elif args.recon_approach== 'fnaf_irim4':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_irim_fnaf_train_1000pixel_val_4x_out_nmse.pickle'
                elif args.recon_approach== 'fnaf_irim8':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_irim_fnaf_train_1000pixel_val_8x_out_nmse.pickle'
                elif args.recon_approach== 'unet4x_on_abnormality_bb':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_bbox_unet_reconstructions_val_4x.pickle'
                elif args.recon_approach== 'unet8x_on_abnormality_bb':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_bbox_unet_reconstructions_val_8x.pickle'
                elif args.recon_approach== 'irim4x_on_abnormality_bb':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_irim_bbox_train_val_4x_out_nmse.pickle'
                elif args.recon_approach== 'irim8x_on_abnormality_bb':
                    dataset_paths['test'] =     'fastMRI/patches_size64_stride8_irim_bbox_train_val_8x_out_nmse.pickle'
                else:
                    print('this dataset is not available')
        else:
            print('this modality is not available')

        dataloaders = patch_anomaly_detection(args, dataset_paths)
        args.class_names = (
            'normal', 'abnormal'
        )  # 0,1 labels
        args.n_channels, args.n_classes = 3, 2
    else:
        NotImplementedError('{} dataset not available.'.format(args.dataset))

    return dataloaders, args


def patch_anomaly_detection(args, dataset_paths):
    """ Loads the patch_anomaly_detection dataset.
        Returns: train/valid/test set split dataloaders.
    """

    config = {'train': True,'valid': True, 'test': True}
    datasets = {i: fastMRIPatchDataset(pickle_file=dataset_paths[i],
                                    transform=transforms.Compose([Standardize(),ToTensor()])) for i in config.keys()}


    if args.distributed:
        config = {'train': None, #DistributedSampler(datasets['train']),
                 'valid': None,
                 'test': None}
    else:
        # weighted sampler weights for training set
        # s_weights = sample_weights(datasets['train'].labels)
        config = {'train': None,# WeightedRandomSampler(s_weights, num_samples=len(s_weights), replacement=True),
                'valid': None,
                'test': None}

    dataloaders = {}
    for i in config.keys():
        if 'val' in i or 'test' in i:
            dataloaders[i] = DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size, shuffle=False)
        else:
            dataloaders[i] = DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size, shuffle=True)
    return dataloaders
