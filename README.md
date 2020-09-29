# Adversarial Robust Training of Deep Learning MRI Reconstruction Models

This repository contains the abnormalities annotation of the knee fastMRI dataset, which were utilized in our paper:<br>["Adversarial Robust Training of Deep Learning MRI Reconstruction Models"](https://pdf), which was recently submitted to the Medical Imaging with Deep Learning (MIDL) 2020 Conference Special Issue at the Machine Learning for Biomedical Imaging (MELBA) Journal.

This work extends the our paper ["Addressing The False Negative Problem of MRI Reconstruction Networks by Adversarial Attacks and Robust Training"](https://2020.midl.io/papers/cheng20.html) by Cheng, K. et al.,2020 which was Awarded Best Paper at MIDL 2020 Conference.

## Abstract
Deep Learning has shown potential in accelerating Magnetic Resonance Image acquisition and reconstruction. Nevertheless, there is a dearth of tailored methods to guarantee that the reconstruction of small features is achieved with high fidelity. In this work, we employ adversarial attacks to generate small synthetic perturbations that when added to the input MRI, they are not reconstructed by a trained DL reconstruction network. Then, we use robust training to increase the network's sensitivity to small features and encourage their reconstruction.
Next, we investigate the generalization of said approach to real world features. For this, a musculoskeletal radiologist annotated a set of cartilage and meniscal lesions from the knee Fast-MRI dataset, and a classification network was devised to assess the features reconstruction. Experimental results show that by introducing robust training to a reconstruction network, the rate (4.8\%) of false negative features in image reconstruction can be reduced. The results are encouraging and highlight the necessity for attention on this problem by the image reconstruction community, as a milestone for the introduction of DL reconstruction in clinical practice. To support further research, we make our annotation publicly available.

## Dataset Annotation
A total of 418 MR exams from the knee Fast-MRI dataset were annotated, 321 and 97 exams from the training and validation sets respectively. The exams included coronal knee sequences, proton density-weighted with and fat suppression. For more details about the acquisition parameters, interested readers are invited to refer to ["Fast-MRI dataset"](https://fastmri.med.nyu.edu/).
Each MRI volumes were examined for cartilage, meniscus, bone marrow lesions (BML), cysts, and based on the Whole Organ Magnetic Resonance Scoring (WORMS) scale,  by the MSK radiologist involved in the study. Bounding boxes were placed for cartilage and BML in 4 sub-regions of the knee at the tibio-femoral joint; medial and lateral femur, medial and lateral tibia. Cartilage lesion were defined as partial or full thickness defect observed in one or more slices extending to include any breadth. Bone marrow lesions annotated included any increased marrow signal abnormality adjacent to articular cartilage in one or more slices, at least encompassing 5\% of the articular marrow region.
Similar bounding boxes were placed in the two sub-regions of meniscus: medial and lateral. Meniscal lesions were defined to include intrasubstance signal abnormality, simple tear, complex tear or maceration. Given the sparse occurrence of the cysts, a single bounding box label was used to encompass all encountered cystic lesions in any of the sub-regions. Fig.~\ref{fig:Example_annotation} is exemplary of the performed annotations. Specifically, the MRI on the left-hand side, displays the presence of cartilage lesions in the medial tibial (red) and femoral (purple) compartments. In the middle a cartilage lesion in the lateral femur is marked by a golden color-coded bounding box. On the right-hand side a map representative of the distribution location of abnormalities in the training set is shown.


In summary the abnormalities annotation follows this naming convention:
'CartMedFem' --> Medial Femoral Cartilage Lesion
'CartLatFem' --> Lateral Femoral Cartilage Lesion
'CartMedTib' --> Medial Tibial Cartilage Lesion
'CartLatTib' --> Lateral Tibial Cartilage Lesion
'BML_Med_Fem' -->Bone Marrow Lesion in Medial Femur subjacent to articular cartilage
'BML_Lat_Fem' -->Bone Marrow Lesion in Lateral Femur subjacent to articular cartilage
'BML_Med_Tib' -->Bone Marrow Lesion in Medial Tibia subjacent to articular cartilage
'BML_Lat_Tib' -->Bone Marrow Lesion in Lateral Tibia subjacent to articular cartilage
'Med_Men' --> Lesion in Medial Meniscus
'Lat_Men' --> Lesion in Lateral Meniscus
'Cyst'  

## Design of the Proposed Model
![model](images/adversarialattack_net.png)

## Annotation example
![annotation](images/example_annotation.png)

## Requirements
- The code has been written in Python (3.6)

## Preparing your data
Please download the ["Fast-MRI dataset"](https://fastmri.med.nyu.edu/) using the official website.
An example of MRI volume from the dataset is added in [dataset](https://github.com/fcaliva/fastMRI_BB_abnormalities_annotation/dataset/singlecoil_val/)

Once downloaded the dataset, please maintain its organization in two folders namely: `singlecoil_train` and `singlecoil_val`.

## Running example code
Note: Set the data path appropriately in `fastMRI_abnormalities_BB_annotation/main.py` before running the code.
To run the code you simply need to use the following script:
```
python main.py --run_example
```
## Running code
Note: <br>
Set the data path appropriately in `fastMRI_abnormalities_BB_annotation/main.py` before running the code.

To run the code you simply need to use the following script:
```
python main.py --fastMRI 'path/to/fastMRI/dataset/' --annotation 'path/to/fastMRI/annotation/' --split 'train or val' --save_in 'path/to/where/to/save/'
```
Example:
```
python main.py --fastMRI '/data/bigbone5/vcheng/fastMRI/datasets/' --annotation 'fastMRI_csv/' --split 'val' --save_in 'BB_png'
```

If you use this annotation for your research, please consider citing our paper.
