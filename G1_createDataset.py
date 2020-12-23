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
    print(flag_test_only)
    exit()

    tmpl_name = "./zeros_PET.nii"
    tmpl_nii = nib.load(tmpl_name)
    tmpl_header = tmpl_nii.header
    tmpl_affine = tmpl_nii.affine

    for folder_name in ["test_gt", "test_f3", "test_recon", "train_recon", "train_gt", "train_f3"]:
            path = "./data/"+name_dataset+"/"+folder_name+"/"
            if not os.path.exists(path):
                os.makedirs(path)

    if flag_test_only:
        tag_list = ["test"]
    else:
        tag_list = ["test", "train"]
    key_list = ["recon", "t1"]
        
    for mat_tag in tag_list:
        for mat_key in key_list:
            tag_key = mat_tag+"_"+mat_key
            mat_list = glob.glob("./nifty/"+name_dataset+"/"+tag_key+"/*.mat")
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

                process_data_DL(mat_data=mat_data, name_dataset=name_dataset,
                                mat_tag=mat_tag, mat_key=mat_key, mat_name=mat_name,
                                tmpl_header=tmpl_header, tmpl_affine=tmpl_affine)

def process_data_DL(mat_data, name_dataset, mat_tag, mat_key, mat_name, tmpl_header, tmpl_affine):

    if mat_key == "recon":
        print("Data dim:", mat_data.shape)
        save_name = "./data/"+name_dataset+"/"+mat_tag+"_"+mat_key+"/"+os.path.basename(mat_name)[:20]+".nii"
        save_file = nib.Nifti1Image(mat_data, affine=tmpl_affine, header=tmpl_header)
        nib.save(save_file, save_name)
        print(save_name)

    if mat_key == "t1":
        px, py, pz = mat_data.shape
        qx, qy, qz = (256, 256, 56)
        zoom_data = zoom(mat_data, (qx/px, qy/py, qz/pz))
        standard_pet = np.zeros((256, 256, 89))
        standard_pet[:, :, 17:73] = zoom_data
        standard_pet[standard_pet<0] = 0

        print("Old dim:", mat_data.shape)
        print("New dim:", standard_pet.shape)
        
        save_file = nib.Nifti1Image(standard_pet, affine=tmpl_affine, header=tmpl_header)
        save_name = "./data/"+name_dataset+"/"+mat_tag+"_gt/"+os.path.basename(mat_name)[:20]+".nii"
        nib.save(save_file, save_name)
        print(save_name)

        smoothed_file = processing.smooth_image(save_file, fwhm=3, mode='nearest')
        save_name = "./data/"+name_dataset+"/"+mat_tag+"_f3/"+os.path.basename(mat_name)[:20]+".nii"
        nib.save(save_file, save_name)
        print(save_name)

if __name__ == "__main__":
    main()

