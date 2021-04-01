# %% codecell
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import h5py
import pickle
import pdb
class fastMRIPatchDataset(Dataset):
    """ patch fastmri dataset. """
    def __init__(self,pickle_file,transform):
        """
        pickle_file (string) including the path to data and annotation
        transform (collable,optional) : optional transforms to be applied on samples
        """
        self.pickle_file = pickle_file
        self.all_samples = pickle.load(open(pickle_file,'rb'))
        self.dataset_size = len(self.all_samples)
        self.transform = transform

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self,idx):
        file_name = self.all_samples[idx][0]
        data_point = h5py.File(file_name,'r')['patch_esc']

        label =  self.all_samples[idx][1]
        sample = {'data':np.array(data_point),'labels': np.array(label)}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        data_point, label = sample['data'], sample['labels']

        min_val = data_point.min() #this can be defined before hand to be the same for the whole dataset
        max_val = data_point.max() #this can be defined before hand to be the same for the whole dataset
        data_point = (data_point - min_val)/(max_val-min_val+1E-10)


        return {'data':np.array(data_point),
                'labels':np.array(label)}

class Standardize(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        data_point, label = sample['data'], sample['labels']

        mean_val = 2.6118338541712485e-05
        std_val  = 2.101291702090038e-05

        data_point = (data_point - mean_val)/(std_val)
        # data_point = (data_point-data_point.min())/(data_point.max()-data_point.min())
        # data_point = (data_point-data_point.mean())/(data_point.std())
        return {'data':np.array(data_point),
                'labels':np.array(label)}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):

        a, label = sample['data'], sample['labels']
        data_point = np.transpose(np.repeat(a[:, :, np.newaxis], 3, axis=2),axes=[2,0,1])
        # data_point = np.transpose(a[:, :, np.newaxis],axes=[2,0,1])

        return {'data':torch.from_numpy(np.array(data_point)),
                'labels':torch.from_numpy(np.array(label))}
