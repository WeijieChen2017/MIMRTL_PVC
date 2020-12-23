from scipy.ndimage import zoom
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel import processing
import glob
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--nameDataset', metavar='', type=str, default="Gibbs",
                        help='Name for the dataset needed to be blur.(Gibbs)<str>') 
    parser.add_argument('--testOnly', action='store_false',
                        help='Whether only search test dataset.(False)<bool>')

    args = parser.parse_args()
    name_dataset = args.nameDataset
    flag_test_only = args.testOnly

    tmpl_name = "./zeros_PET.nii"
    tmpl_nii = nib.load(tmpl_name)
    tmpl_header = tmpl_nii.header
    tmpl_affine = tmpl_nii.affine

    for folder_name in ["gt", "gt_f3", "blur", "test", "pure"]:
            path = "./data/"+name_dataset+"/"+folder_name+"/"
            if not os.path.exists(path):
                os.makedirs(path)

    key_list = ["test"]
    if not flag_test_only:
        key_list.append("t1")
        key_list.append("recon")
        
    for mat_key in key_list:
        mat_list = glob.glob("./nifty/"+name_dataset+"/"+mat_key+"/*.mat")
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
                process_data_t1_DL(mat_data, name_dataset)
            if mat_key == "recon":
                process_data_recon_DL(mat_data, name_dataset)
            if mat_key == "test":
                process_data_test_DL(mat_data, name_dataset)



def process_data_recon_DL(data, name_dataset):
    zoom_data = data
    print("Old dim:", data.shape)
    print("New dim:", zoom_data.shape)
    
    save_file = nib.Nifti1Image(zoom_data, affine=tmpl_affine, header=tmpl_header)
    save_name = "./data/"+name_dataset+"/blur/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(save_file, save_name)
    print(save_name)

def process_data_t1_DL(data, name_dataset):
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
    save_name = "./data/"+name_dataset+"/pure/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(smoothed_file, save_name)
    print(save_name)


def process_data_test_DL(data, name_dataset):

    zoom_data = data   
    save_file = nib.Nifti1Image(zoom_data, affine=tmpl_affine, header=tmpl_header)
    save_name = "./data/"+name_dataset+"/test/"+os.path.basename(mat_name)[:20]+".nii"
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
    save_name = "./data/"+name_dataset+"/gt/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(save_file, save_name)
    print(save_name)

    smoothed_file = processing.smooth_image(save_file, fwhm=3, mode='nearest')
    save_name = "./data/"+name_dataset+"/gt_f3/"+os.path.basename(mat_name)[:20]+".nii"
    nib.save(smoothed_file, save_name)
    print(save_name)

if __name__ == "__main__":
    main()

