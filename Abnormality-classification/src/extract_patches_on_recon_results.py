import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
from skimage import measure
import pickle

split = 'val' # options available ['train', 'val']
if split not in ['train','val']:
    print('wrong split specified - possible options are ["train","val"]')
    exit()

if split == 'train':
    max_r, max_c, max_z = 0, 0, 0
elif split == 'val':
    max_r, max_c, max_z = 0, 0, 0

'''
[max_c, max_r, max_z] are computed using the abnormality annotation. 
These represent the location in the training set where to extract the patches. 
These values are used to reposition the patch in the actual fullimage space and extract the patch
'''

general_save_folder = 'fastMRI/extracted_patches'
save_pickle_in = '../pickle_files/'
if general_save_folder[-1] != '/':
    general_save_folder+='/'

if not os.path.isdir(general_save_folder):
    os.makedirs(general_save_folder)

for patch_size, stride in zip([32,64],[2,8]):
    print(f'Now generating patches size {patch_size}x{patch_size} with stride {stride} on split {split}')

    # path to the pickle file which 
    pickle_patches = pickle.load(open(f'../pickle_files/patches_size{patch_size}_stride{stride}_{split}.pickle','rb'))
    
    # paths to the folder which includes the reconstructed MRIs. These should be saved with the same convention as the fastmri dataset.
    # an example file is included in the folder data_sample. sample_data/file1000017.h5
 
    all_path_modality = ['h5_recons/unet4x/',
                        'h5_recons/fnaf-unet4x/',
                        'h5_recons/bbox-unet4x',
                        'h5_recons/irim4x/',
                        'h5_recons/fnaf-irim4x',
                        'h5_recons/bbox-irim4x',
                        'h5_recons/unet8x/',                        
                        'h5_recons/fnaf-unet8x/',  
                        'h5_recons/bbox-unet8x',                     
                        'h5_recons/irim8x/',                        
                        'h5_recons/fnaf-irim8x',
                        'h5_recons/bbox-irim8x']
                        
    modality = [x.split('/')[-2] for x in all_path_modality]

    for id_mod in range(len(all_path_modality)):
        predictions_path = all_path_modality[id_mod]
        pickle_cnt = []
        save_patches_in = f'{general_save_folder}patches_size{patch_size}_stride{stride}_{modality[id_mod]}/'        
        try:
            os.makedirs(save_patches_in)
            print(f'output will be saved in {save_patches_in}')
        except:
            print(f'output will be saved in {save_patches_in}')

        for id_file in range(len(pickle_patches)):
            file = pickle_patches[id_file]
            label = file[1]
            name = file[0].split('/')[-1].split('.')[0].split('_')[0]
            yy = int(file[0].split('/')[-1].split('.')[0].split('_r')[-1].split('_')[0])
            xx = int(file[0].split('/')[-1].split('.')[0].split('_c')[-1].split('_')[0])
            zz = int(file[0].split('/')[-1].split('.')[0].split('_z')[-1].split('_')[0])

            big_mri_esc = np.zeros([max_r,max_c,max_z])
            mri_recon = np.transpose(np.flip(np.array(h5py.File((predictions_path+name+'.h5'),'r')['reconstruction']),axis=1),[1,2,0])
            r,c,z = mri_recon.shape
            cropping_required_uplrba = [int(np.floor(max_r-r)/2),int(np.ceil(max_r-r)/2),int(np.floor(max_c-c)/2),int(np.ceil(max_c-c)/2),int(np.floor(max_z-z)/2),int(np.ceil(max_z-z)/2)]
            big_mri_esc[cropping_required_uplrba[0]:cropping_required_uplrba[0]+r,cropping_required_uplrba[1]:cropping_required_uplrba[1]+c,cropping_required_uplrba[2]:cropping_required_uplrba[2]+z] = mri_recon

            # This loads the original fastMRI sample
            # hf_fastmri = np.array(h5py.File(file[0],'r')['patch_esc'])
            
            hf_recon = big_mri_esc[yy:yy+patch_size,xx:xx+patch_size,zz]
            
            name = file[0].split('/')[-1]
            filename = save_patches_in + name
            hf = h5py.File(filename,'w')
            hf.create_dataset('patch_esc', data = hf_recon.astype(np.float32))
            hf.close()
            pickle_cnt.append([filename,label])
        pickle.dump(pickle_cnt, open(
            f'{save_pickle_in}patches_size{patch_size}_stride{stride}_{modality[id_mod]}_{split}.pickle', 'wb'))
