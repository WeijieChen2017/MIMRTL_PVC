import os
import glob
import cv2
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import nibabel
from PIL import Image
from nibabel import processing
from skimage.transform import radon, iradon


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--nameDataset', metavar='', type=str, default="sk8R",
                        help='Name for the dataset needed to be blur.(sk8R)<str>') 
    parser.add_argument('--fwhmHub', metavar='', type=str, default="8",
                        help='Blur kernels for sythesizing.(8)<str>')
    parser.add_argument('--gauHub', metavar='', type=str, default="1e-3, 5e-3",
                        help='Gaussian noise level for sythesizing.(1e-3, 5e-3)<str>')
    parser.add_argument('--poiHub', metavar='', type=str, default="1, 5",
                        help='Poisson noise level for sythesizing.(1, 5)<str>')
    parser.add_argument('--radon', metavar='', type=bool, default=False,
                        help='Whether add radon artifacts for sythesizing.(False)<bool>')
    parser.add_argument('--theta', metavar='', type=int, default="112",
                        help='Unit angle for sythesizing.(112)<int>')


    # fwhm_hub = [8]
    # gau_sigma_hub = [1e-3,3e-3,5e-3,7e-3,9e-3]
    # poi_sigma_hub = [1,3,5,7,9]
    # gau_sigma_hub=[1e-3, 5e-3]
    # poi_sigma_hub=[1, 5]
    # gau_sigma_hub=[1*1e-2,3*1e-2]
    # poi_sigma_hub=[1e2]
    # gau_sigma_hub=[]
    # poi_sigma_hub=[]
    # flag_Radon = False
    # fwhm_hub = [0, 0.5, 1, 1.5, 2, 2.5]
    # theta = np.linspace(0., 360., 28*4, endpoint=False) # max(image.shape)
    args = parser.parse_args()
    name_dataset = args.nameDataset
    fwhm_hub = list(map(int, args.fwhmHub.split(sep=',')))
    gau_sigma_hub = list(map(float, args.gauHub.split(sep=',')))
    poi_sigma_hub = list(map(float, args.poiHub.split(sep=',')))

    if gau_sigma_hub == [0.0]:
        gau_sigma_hub = []
    if poi_sigma_hub == [0.0]:
        poi_sigma_hub = []

    flag_Radon = args.radon
    theta = np.linspace(0., 360., args.theta, endpoint=False)
    print("Dataset name:", name_dataset)
    print("Fwhm hub: ", fwhm_hub)
    print("Gau noise: ", gau_sigma_hub)
    print("Poi noise: ", poi_sigma_hub)
    print("Radon: ", flag_Radon)

    sythesize_data(name_dataset=name_dataset, fwhm_hub=fwhm_hub,
                   gau_sigma_hub=gau_sigma_hub, poi_sigma_hub=poi_sigma_hub,
                   flag_Radon=flag_Radon, theta=theta)


def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data


def nib_smooth(file_mri, data, fwhm, tag, save_path):
    nii_file = nibabel.Nifti1Image(data, file_mri.affine, file_mri.header)
    smoothed = processing.smooth_image(nii_file, fwhm=fwhm, mode='nearest')
    smoothed_data = maxmin_norm(np.asanyarray(smoothed.dataobj))
    smoothed_file = nibabel.Nifti1Image(smoothed_data, file_mri.affine, file_mri.header)
#     print(np.amax(smoothed_file.get_fdata()))
    nibabel.save(smoothed_file, save_path+"fwhm_"+str(fwhm)+"_"+tag+".nii")
    print("fwhm_"+str(fwhm)+"_"+tag+".nii")

def sythesize_data(name_dataset='sk8R', fwhm_hub=[8],
                   gau_sigma_hub=[], poi_sigma_hub=[], flag_Radon=False, theta=[]):

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

    list_ori = glob.glob(pure_path+"*.nii")
    list_ori.sort()
    for path_ori in list_ori:
        print(path_ori)
        file_mri = nibabel.load(path_ori)
        data_mri = np.asanyarray(file_mri.dataobj)
        file_name = os.path.basename(path_ori)
    #     nibabel.save(file_mri, pure_path+file_name)
        print(data_mri.shape)

        for idx_fwhm in fwhm_hub:
            tag = file_name[:-4]+""
            nib_smooth(file_mri, data_mri, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

            # gaussian noise
            for idx_gau_sigma in gau_sigma_hub:
                noise = np.random.normal(0, idx_gau_sigma*np.var(data_mri), data_mri.shape)
                noisy_img = data_mri + noise
                tag = file_name[:-4]+"_gs_"+'{:.0e}'.format(idx_gau_sigma)
                nib_smooth(file_mri, noisy_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

            # poisson noise
            for idx_poi_sigma in poi_sigma_hub:
                noise = np.random.poisson(size=data_mri.shape, lam=np.mean(data_mri)*idx_poi_sigma)
                noisy_img = data_mri + noise
                tag = file_name[:-4]+"_ps_"+'{:.0e}'.format(idx_poi_sigma)
                nib_smooth(file_mri, noisy_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)
        
        if flag_Radon:
            # radon transform, https://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.html
            radon_img = np.zeros(data_mri.shape)
            for idx_slice in range(data_mri.shape[2]):
                orginal_img = data_mri[:, :, idx_slice]
                sinogram = radon(orginal_img, theta=theta, circle=False)
                reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
                radon_img[:, :, idx_slice] = reconstruction_fbp

            for idx_fwhm in fwhm_hub:
                tag = file_name[:-4]+"_radon"
                nib_smooth(file_mri, radon_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

                # gaussian noise
                for idx_gau_sigma in gau_sigma_hub:
                    noise = np.random.normal(0, idx_gau_sigma*np.var(data_mri), data_mri.shape)
                    noisy_img = radon_img + noise
                    tag = file_name[:-4]+"_radon_gs_"+'{:.0e}'.format(idx_gau_sigma)
                    nib_smooth(file_mri, noisy_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

                # poisson noise
                for idx_poi_sigma in poi_sigma_hub:
                    noise = np.random.poisson(size=data_mri.shape, lam=np.mean(data_mri)*idx_poi_sigma)
                    noisy_img = radon_img + noise
                    tag = file_name[:-4]+"_radon_ps_"+'{:.0e}'.format(idx_poi_sigma)
                    nib_smooth(file_mri, noisy_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

if __name__ == "__main__":
    main()


