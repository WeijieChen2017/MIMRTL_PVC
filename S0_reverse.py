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

import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--nameDataset', metavar='', type=str, default="sk8R",
                        help='Name for the dataset needed to be reverse.(2dEnhancedSk8)<str>')
    parser.add_argument('--mriFolder', metavar='', type=str, default="mri",
                        help='Name for the dataset needed to be reverse.(2dEnhancedSk8)<str>')
    parser.add_argument('--powerFactor', metavar='', type=float, default=1,
                        help='Contrast enhancing factor.(0.5)<float>')

    args = parser.parse_args()
    name_dataset = args.nameDataset
    power_factor = args.powerFactor
    mri_folder = args.mriFolder
    reverse_dataset(name_dataset, mri_folder, power_factor)

def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

def reverse_dataset(name_dataset, mri_folder, powerFactor):
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

    test_path = "./data/"+name_dataset+"/test/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    mri_path = "./data/"+mri_folder+"/"
    if not os.path.exists(mri_path):
        os.makedirs(mri_path)

    list_ori = glob.glob(mri_path+"*.nii")
    list_ori.sort()
    for path_ori in list_ori:
        print(path_ori)
        nii_name = os.path.basename(path_ori)[:-7]
        nii_file = nib.load(path_ori)
        data_mri = np.asanyarray(nii_file.dataobj)
        
        # otsu_data = np.zeros(data_mri.shape)
        # for idx in range(data_mri.shape[2]):
        #     if idx >=0 and idx < 225:
        #         img = data_mri[:, :, idx]    
        #         otsu_data[:, :, idx] = img >= threshold_otsu(img)
        
        # reverse
        norm_mri = 1-maxmin_norm(data_mri)
        norm_mri[norm_mri == 1] = 0

        # enhance
        # power_hub = [0.5,1,2,3]
        # for power in power_hub:
        norm_mri_p = norm_mri ** powerFactor
        file_inv = nib.Nifti1Image(norm_mri_p, nii_file.affine, nii_file.header)
        save_name = pure_path+"/"+nii_name+"_inv_mask_p"+str(powerFactor)+".nii"
        nib.save(file_inv, save_name)
            
        print(save_name)
        # norm_mri[otsu_data>0] = 255-norm_mri[otsu_data>0]
        
        # cut_th_0 = 100
        # cut_point_0 = np.percentile(norm_mri, cut_th_0)
    #     cut_point_1 = np.percentile(norm_mri, 99.9)
        
        # norm_mri[norm_mri < cut_point_0] = cut_point_0
    #     norm_mri[norm_mri > cut_point_1] = cut_point_1
        # norm_mri = maxmin_norm(norm_mri)*255
        
        
        

if __name__ == "__main__":
    main()

