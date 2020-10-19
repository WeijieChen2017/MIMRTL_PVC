from scipy.ndimage import zoom
from scipy.io import loadmat
import numpy as np
import nibabel as nib
import glob
import os

folder_list = glob.glob("./oct19_*/")
for folder_name in folder_list:
    print("-------------------")
    print(folder_name)

    pure_dir = folder_name+"/pure/"
    blur_dir = folder_name+"/blur/"
    test_dir = folder_name+"/test/"

    for subdir in [pure_dir, blur_dir, test_dir]:
        nii_list = glob.glob(subdir+"*.nii")
        for nii_dir in nii_list:
            print(nii_dir)
            nii_file = nib.load(nii_dir)
            nii_data = nii_file.get_fdata()

            file_header = nii_file.header
            file_affine = nii_file.affine

            # data[data<0] = 0
            # data[data>1] = 1

            px, py, pz = nii_data.shape
            qx, qy, qz = (512, 512, 89)
            if (px != qx) or (py != qy) or (pz != qz)
                zoom_data = zoom(nii_data, (qx/px, qy/py, qz/pz))

                print("Old dim:", nii_data.shape)
                print("New dim:", zoom_data.shape)

                new_file = nib.Nifti1Image(zoom_data, affine=file_affine, header=file_header)
                nib.save(new_file, nii_dir)
            else:
                print("Old dim:", nii_data.shape)
    
    print("-------------------")
