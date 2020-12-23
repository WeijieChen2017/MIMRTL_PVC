import os
import glob
import cv2
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import zoom


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--nameDataset', metavar='', type=str, default="sk8R",
                        help='Name for the dataset needed to be reverse.(sk8R)<str>')
    parser.add_argument('--dataFolder', metavar='', type=str, default="pMix",
                        help='Folder of testing dataset.(unet)<str>')
    parser.add_argument('--outputChannel', metavar='', type=int, default=1,
                        help='Output channel of training dataset.(7)<int>')
    parser.add_argument('--resizeFactor', metavar='', type=int, default=1,
                        help='Resizing factor of training dataset.(1)<int>')

    args = parser.parse_args()
    name_dataset = args.nameDataset
    data_folder = args.dataFolder
    output_chan = args.outputChannel
    resize_f = args.resizeFactor

    print("name_dataset=", name_dataset)
    print("data_folder=", data_folder)
    print("output_chan=", output_chan)
    print("resize_f=", resize_f)

    assemble_results(name_dataset=name_dataset, data_folder=data_folder,
                     output_chan=output_chan, resize_f=resize_f)


def assemble_results(name_dataset="sk8R", data_folder="pMix", output_chan=7, resize_f=1):

    syn_path = "./data/"+data_folder+"/test_syn/"
    if not os.path.exists(syn_path):
        os.makedirs(syn_path)

    list_X = glob.glob("./data/"+data_folder+"/test_recon/*.nii")
    # list_ori = glob.glob("./data/sk8R_721/test/*.nii")
    list_X.sort()
    for path_X in list_X:
        print(path_X)
        nii_name = os.path.basename(path_X)[:-4]
        nii_file = nib.load(path_X)
        nii_data = np.asanyarray(nii_file.dataobj)
        qx, qy, qz = nii_data.shape
        
        
    #     pred_real = np.zeros((nii_data.shape[0], nii_data.shape[1], nii_data.shape[2]))
        pred_fake = np.zeros((qx, qy, qz))
        
        for idx in range(nii_data.shape[2]):
            path_fake = "./pytorch-CycleGAN-and-pix2pix/results/"+name_dataset+"/test_latest/images/"+nii_name+"_"+str(idx)+"_fake.npy"
    #         img = cv2.resize(np.asarray(plt.imread(path_fake)), dsize=(nii_data.shape[0], nii_data.shape[1]), interpolation=cv2.INTER_CUBIC)
            img = np.squeeze(np.load(path_fake))
            
            # img[img<0] = 0
            if output_chan == 1:
                px, py = img.shape
                img = zoom(img, (qx/px, qy/py))
            else:
                pz, px, py = img.shape
                img = zoom(img[int(output_chan//2), :, :], (qx/px, qy/py))
            pred_fake[:, :, idx] = zoom(img, zoom=1/resize_f)
        
        factor_f = np.sum(nii_data)/np.sum(pred_fake)
        file_fake = nib.Nifti1Image(pred_fake*factor_f, nii_file.affine, nii_file.header)
        nib.save(file_fake, syn_path+nii_name+"_"+name_dataset+".nii")
    print("------------------------------------------------------------")
    print("----------------------Finished------------------------------")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
