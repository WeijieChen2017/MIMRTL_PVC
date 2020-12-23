from scipy.ndimage import zoom
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel import processing
import glob
import os

def process_data_recon_DL(data):
    zoom_data = data
    print("Old dim:", data.shape)
    print("New dim:", zoom_data.shape)
    
    save_file = nib.Nifti1Image(zoom_data, affine=tmpl_affine, header=tmpl_header)
    save_name = "./data/Gibbs/blur/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(save_file, save_name)
    print(save_name)

def process_data_t1_DL(data):
    px, py, pz = data.shape
    qx, qy, qz = (256, 256, 56)
    zoom_data = zoom(data, (qx/px, qy/py, qz/pz))
    standard_pet = np.zeros((256, 256, 89))
    standard_pet[:, :, 17:73] = zoom_data
    standard_pet[standard_pet<0] = 0

    print("Old dim:", data.shape)
    print("New dim:", standard_pet.shape)
    
    save_file = nib.Nifti1Image(standard_pet, affine=tmpl_affine, header=tmpl_header)  
    smoothed_file = processing.smooth_image(save_file, fwhm=3, mode='nearest')
    save_name = "./data/Gibbs/pure/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(smoothed_file, save_name)
    print(save_name)


def process_data_test_DL(data):

	zoom_data = data   
    save_file = nib.Nifti1Image(zoom_data, affine=tmpl_affine, header=tmpl_header)
    save_name = "./data/Gibbs/test/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(save_file, save_name)
    print(save_name)

    px, py, pz = data.shape
    qx, qy, qz = (256, 256, 56)
    zoom_data = zoom(data, (qx/px, qy/py, qz/pz))
    standard_pet = np.zeros((256, 256, 89))
    standard_pet[:, :, 17:73] = zoom_data
    standard_pet[standard_pet<0] = 0

    print("Old dim:", data.shape)
    print("New dim:", standard_pet.shape)
    
    save_file = nib.Nifti1Image(standard_pet, affine=tmpl_affine, header=tmpl_header)
    save_name = "./data/Gibbs/gt/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(save_file, save_name)
    print(save_name)

    smoothed_file = processing.smooth_image(save_file, fwhm=3, mode='nearest')
    save_name = "./data/Gibbs/gt_f3/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(smoothed_file, save_name)
    print(save_name)

tmpl_name = "./test_data/zeros_PET.nii"
tmpl_nii = nib.load(tmpl_name)
tmpl_header = tmpl_nii.header
tmpl_affine = tmpl_nii.affine

for folder_name in ["gt", "gt_f3", "blur", "test", "pure"]:
        path = "./data/Gibbs/"+folder_name+"/"
        if not os.path.exists(path):
            os.makedirs(path)

for mat_key in ["test"]:
	mat_list = glob.glob("./"+mat_key+"/*.mat")
	mat_list.sort()
	for mat_name in mat_list:
	    print("-----------------------------------------------")
	    mdict = loadmat(mat_name)

	    try:
	        mat_data = mdict["reconImg"]
	    except Exception:
	        pass  # or you could use 'continue'

	    try:
	        mat_data = mdict["data"]
	    except Exception:
	        pass  # or you could use 'continue'

	#     print(mat_data.shape)
	    if mat_key == "t1":
	        process_data_t1_DL(mat_data)
	    if mat_key == "recon":
	        process_data_recon_DL(mat_data)
	    if mat_key == "test":
	    	process_data_test_DL(mat_data)

