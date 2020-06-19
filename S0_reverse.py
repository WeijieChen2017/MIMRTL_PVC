import os
import glob
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import nibabel
from skimage.transform import radon, iradon
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank

def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

name_dataset = "2d_enhanced_sk8"

for folder_name in ["trainA", "trainB", "testA", "testB"]:
    path = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/"+folder_name+"/"
    if not os.path.exists(path):
        os.makedirs(path)
        
blur_path = "./data/"+name_dataset+"/blur/"
if not os.path.exists(blur_path):
    os.makedirs(blur_path)
    
pure_path = "./data/"+name_dataset+"/pure/"
if not os.path.exists(pure_path):
    os.makedirs(pure_path)

list_ori = glob.glob("./data/"+name_dataset+"/mri/*.nii")
list_ori.sort()
for path_ori in list_ori:
    print(path_ori)
    nii_name = os.path.basename(path_ori)[:-7]
    nii_file = nib.load(path_ori)
    data_mri = np.asanyarray(nii_file.dataobj)
    
    otsu_data = np.zeros(data_mri.shape)
    for idx in range(data_mri.shape[2]):
        if idx >=0 and idx < 225:
            img = data_mri[:, :, idx]    
            otsu_data[:, :, idx] = img >= threshold_otsu(img)
    
    # reverse
    norm_mri = maxmin_norm(data_mri)*255
    norm_mri[otsu_data>0] = 255-norm_mri[otsu_data>0]
    
    cut_th_0 = 80
    cut_point_0 = np.percentile(norm_mri, cut_th_0)
#     cut_point_1 = np.percentile(norm_mri, 99.9)
    
    norm_mri[norm_mri < cut_point_0] = cut_point_0
#     norm_mri[norm_mri > cut_point_1] = cut_point_1
    norm_mri = maxmin_norm(norm_mri)*255
    
    
    file_inv = nib.Nifti1Image(norm_mri, nii_file.affine, nii_file.header)
    save_name = pure_path+"/"+nii_name+"_inv_mask_p"+str(cut_th_0)+".nii"
    nib.save(file_inv, save_name)
    
    print(save_name)

