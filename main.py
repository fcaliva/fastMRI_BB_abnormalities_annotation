import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--fastMRI', default='dataset/',
                    help='path/to/fastMRI/dataset/')
PARSER.add_argument('--annotation', default='fastMRI_csv/',
                    help='path/to/fastMRI/annotation/')
PARSER.add_argument('--split', default='val',
                    help='split: options are train val')
PARSER.add_argument('--save_png', dest='save_png', action='store_false',
                    help='choose whether or not to save PNG files.')
PARSER.set_defaults(save_png=True)
PARSER.add_argument('--display_on_screen', dest='display_on_screen', action='store_true',
                    help='choose whether or not to display annotation on screed.')
PARSER.set_defaults(display_on_screen=False)
PARSER.add_argument('--run_example', dest='run_example', action='store_true',
                    help='choose whether or not to run the example.')
PARSER.set_defaults(run_example=False)
PARSER.add_argument('--save_in', default='BB_png',
                    help='path/to/where/to/save/')

args = PARSER.parse_known_args()[0]
## load the annotation
split = args.split
if args.save_png:
    where_to_save = f'{args.save_in}/{split}/'
    try:
        print(f'PNG files will be saved in {where_to_save}')
        os.makedirs(where_to_save)
    except:
        1
path = f'{args.annotation}singlecoil_bb_{split}.csv'
csv_file = pd.read_csv(path)
csv_file.head()

filenames = csv_file['filename'].unique()

labels = ['CartMedFem','CartLatFem','CartMedTib','CartLatTib','BML_Med_Fem','BML_Lat_Fem','BML_Med_Tib','BML_Lat_Tib','Med_Men','Lat_Men','Cyst']
print(f"There are {len(labels)} annotation labels. \nThey are: \n{labels}\n\n")
labelsToIdx = {value:idx for idx,value in enumerate(labels)}
colors = ['r','r','c','c','m','m','w','w','y','y','b']

# extract annotation for MRI idx
if args.run_example:
    show_vol =[1]
else:
    show_vol =range(len(filenames))
for idx in show_vol:
    print(f"Inspecting file: {filenames[idx]}")

    # collect annotation for that MRI
    annotation = csv_file[csv_file['filename']==filenames[idx]]
    print(annotation)

    # define path where the fastmri dataset is stored
    path_to_fast_mri_dataset = f'{args.fastMRI}singlecoil_{split}/'

    # load MRI volume from fastMRI dataset
    MRI = np.flip(np.array(h5py.File(path_to_fast_mri_dataset+filenames[idx]+'.h5','r')['reconstruction_esc']),axis=1)

    # recall that the annotation ranges from slice 1 to Nslices,
    # so in python need to subtract 1.
    # Coordinates of the bounding-boxes are specified in the range [0,numPixels]
    # collect id slices where any annotation is present
    slices = annotation['slice'].unique()
    for idx_sl in range(len(slices)):
        sl = slices[idx_sl]
        # collect annotation in the particular slice
        slice_annotation = annotation[annotation['slice']==sl]
        n_bbx = slice_annotation['numBBs'].item()
        bbx_labels = np.array(slice_annotation['labels'])
        if n_bbx > 1:
            bb_lab = [x for x in bbx_labels.item().replace('[\'','').replace('\']','').replace('\' \'','\'').split('\'')]
        else:
            bb_lab = [bbx_labels.item().replace('[\'','').replace('\']','').replace('\' \'','\'')]
        bbx_coord =  slice_annotation['coords']
        print(f"There are {n_bbx} bbx, in slice {sl}. \nThey are {bbx_labels.item()}.")

        bb_c = [x for x in bbx_coord.item().replace('[[','').replace(']]',']').replace('\n [','').split(']')[:-1]]
        fig, ax = plt.subplots(1)
        fig.set_figheight(10);fig.set_figwidth(10);
        for ii,bb in enumerate(bb_c):
            xy = [np.float32(x) for x in bb.split(' ') if len(x)>0]
            width = xy[1]-xy[0]
            height = xy[3]-xy[2]
            rect= patches.Rectangle((xy[0],xy[2]), width, height, linewidth = 2, edgecolor= colors[labelsToIdx[bb_lab[ii]]], facecolor = 'none')
            img = MRI[sl-1]
            ax.imshow(img, cmap = 'gray')
            ax.add_patch(rect); plt.axis('off')
            plt.title(f"{filenames[idx]}, BB: {bb_lab[ii]}, SL {sl}", fontsize=18)
        if args.save_png:
            plt.savefig(f'{where_to_save}{filenames[idx]}_{sl}.png',dpi=400, bbox_inches = 'tight')
        if args.display_on_screen:
            plt.show()
