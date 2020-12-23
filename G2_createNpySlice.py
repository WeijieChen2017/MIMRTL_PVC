import os
import glob
import cv2
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from PIL import Image

def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--nameDataset', metavar='', type=str, default="sk8R",
                        help='Name for the dataset needed to be reverse.(sk8R)<str>')
    parser.add_argument('--nameModel', metavar='', type=str, default="unet",
                        help='Name of training model.(unet)<str>')
    parser.add_argument('--inputChannel', metavar='', type=int, default=7,
                        help='Input channel of training dataset.(7)<int>')
    parser.add_argument('--outputChannel', metavar='', type=int, default=1,
                        help='Output channel of training dataset.(7)<int>')
    parser.add_argument('--resizeFactor', metavar='', type=int, default=1,
                        help='Resizing factor of training dataset.(1)<int>')
    parser.add_argument('--norm', metavar='', type=str, default="maxmin",
                        help='Normalization for training.(maxmin)<str>')
    parser.add_argument('--save', metavar='', type=bool, default=True,
                        help='Whether to save images.(maxmin)<str>')
    parser.add_argument('--testOnly', action='store_true',
                    help='Whether only search test dataset.(store_true)<bool>')



    args = parser.parse_args()
    name_dataset = args.nameDataset
    name_model = args.nameModel
    input_chan = args.inputChannel
    output_chan = args.outputChannel
    resize_f = args.resizeFactor
    save_flag = args.save
    norm = args.norm
    flag_test_only = args.testOnly

    print("name_dataset=", name_dataset)
    print("name_model=", name_model)
    print("input_chan=", input_chan)
    print("output_chan=", output_chan)
    print("resize_f=", resize_f)
    print("normalization=", norm)
    print("save_flag=", save_flag)

    create_dataset(name_dataset=name_dataset, name_model = name_model,
                   input_chan=input_chan, output_chan=output_chan, resize_f=resize_f,
                   norm=norm, save_flag=save_flag, test_only=flag_test_only)



def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    if MAX-MIN > 0:
        data = (data - MIN)/(MAX-MIN)
    return data

def z_norm(data):
    MEAN = np.mean(data)
    STD = np.std(data)
    return (data - MEAN) / STD

def create_index(dataA, n_slice, zeroPadding=False):
    h, w, z = dataA.shape
    index = np.zeros((z,n_slice))
    
    for idx_z in range(z):
        for idx_c in range(n_slice):
            index[idx_z, idx_c] = idx_z-(n_slice-idx_c+1)+n_slice//2+2
    if zeroPadding:
        index[index<0]=z
        index[index>z-1]=z
    else:
        index[index<0]=0
        index[index>z-1]=z-1
    return index

def slice5_AB(dataA, dataB, name_dataset, n_slice=1, name_tag="", resize_f=1):
    # shape supposed to be 512*512*284 by default
    assert dataA.shape == dataB.shape, ("DataA should share the same shape with DataB.")
    path2save = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/train/"
    h, w, z = dataA.shape
    h = h*resize_f
    w = w*resize_f
    img = np.zeros((n_slice, h, w*2))
    index = create_index(dataA, n_slice)
        
    for idx_z in range(z):
        for idx_c in range(n_slice):
            img[idx_c, :, :w] = zoom(dataA[:, :, int(index[idx_z, idx_c])], zoom=resize_f)
            img[idx_c, :, w:] = zoom(dataB[:, :, int(index[idx_z, idx_c])], zoom=resize_f)
        name2save = path2save+name_tag+"_"+str(idx_z)+".npy"
        np.save(name2save, img)
    print(str(z)+" images have been saved.")

def slice5_A(dataA, name_dataset, n_slice=1, name_tag="", resize_f=1, folderName='test'):
    # shape supposed to be 512*512*284 by default
    path2save = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+'/'+folderName+'/'
    h, w, z = dataA.shape
    h = h*resize_f
    w = w*resize_f
    img = np.zeros((n_slice, h, w))
    index = create_index(dataA, n_slice, zeroPadding=False)
    print(dataA.shape)
        
    for idx_z in range(z):
        for idx_c in range(n_slice):
            img[idx_c, :, :] = zoom(dataA[:, :, int(index[idx_z, idx_c])], zoom=resize_f)
        name2save = path2save+name_tag+"_"+str(idx_z)+".npy"
        np.save(name2save, img)
    print(str(z)+" images have been saved.")

def createAB(dataA, dataB, name_dataset, chanA=7, chanB=1,
             name_tag="", resize_f=1, folderName='test', save_flag=False):
    # shape supposed to be 512*512*284 by default
    assert dataA.shape == dataB.shape, ("DataA should share the same shape with DataB.")
    path2save = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/train/"
    h, w, z = dataA.shape
    h = h*resize_f
    w = w*resize_f
    n_save = 0
    
    imgA = np.zeros((chanA, h, w))
    imgB = np.zeros((chanB, h, w))
    indexA = create_index(dataA, chanA)
    indexB = create_index(dataB, chanB)

    for idx_z in range(z):

        for idx_a in range(chanA):
            imgA[idx_a, :, :] = zoom(dataA[:, :, int(indexA[idx_z, idx_a])], zoom=resize_f)
        for idx_b in range(chanB):
            imgB[idx_b, :, :] = zoom(dataB[:, :, int(indexB[idx_z, idx_b])], zoom=resize_f)

        imgA = maxmin_norm(imgA)
        imgB = maxmin_norm(imgB)

        img = [imgA, imgB]
        if np.amax(imgB)-np.amin(imgB) > 0:
            name2save = path2save+name_tag+"_"+str(idx_z)+".npy"
            n_save += 1
        # if save_flag:
            np.save(name2save, img)
    print(str(n_save)+" images have been saved.")


def create_dataset(name_dataset='sk8R', name_model = "unet", input_chan=7, output_chan=7,
                   resize_f=1, norm="maxmin", save_flag=False, test_only=False):

    for folder_name in ["train", "test", "trainA", "trainB", "testA", "testB"]:
        path = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/"+folder_name+"/"
        if not os.path.exists(path):
            os.makedirs(path)

    if name_model == 'cGAN':
        sliceUsed = slice5_AB
    if name_model == 'unet':
        sliceUsed = createAB

    if norm == "maxmin":
        normUsed = maxmin_norm
    if norm == 'znorm':
        normUsed = z_norm

    list_X = glob.glob("./data/"+name_dataset+"/test_recon/*.nii")
    list_X.sort()
    print("Test:")
    for path_X in list_X:
        filename_X = os.path.basename(path_X)[:-4]
        print(filename_X)
        data_X = normUsed(nib.load(path_X).get_fdata())
        px, py, pz = data_X.shape
        qx, qy, qz = (256, 256, pz)
        zoom_data_X = zoom(data_X, (qx/px, qy/py, qz/pz))
        print("test X shape: ", zoom_data_X.shape)

        slice5_A(dataA=zoom_data_X, name_dataset=name_dataset, n_slice=input_chan,
                  name_tag=filename_X, resize_f = resize_f, folderName='test')
        print("------------------------------------------------------------------------")

    if not test_only:
        list_X = glob.glob("./data/"+name_dataset+"/train_recon/*.nii")
        list_X.sort()
        print("Train:")
        for path_X in list_X:
            filename_X = os.path.basename(path_X)[:-4]
            print(filename_X)
            data_X = normUsed(nib.load(path_X).get_fdata())
            px, py, pz = data_ori.shape
            qx, qy, qz = (256, 256, pz)
            zoom_data_X = zoom(data_X, (qx/px, qy/py, qz/pz))
            print("train X shape: ", zoom_data_X.shape)
            
            list_Y = glob.glob("./data/"+name_dataset+"/train_f3/*"+filename_X+"*.nii")
            list_Y.sort()
            
            for path_Y in list_Y:
                print("Pairs")
                filename_Y = os.path.basename(path_Y)[:-4]
                print("X:", filename_X)
                print("Y:", filename_Y)
                data_Y = normUsed(nib.load(path_Y).get_fdata())
                px, py, pz = data_Y.shape
                qx, qy, qz = (256, 256, pz)
                zoom_data_Y = zoom(data_Y, (qx/px, qy/py, qz/pz))
                print("train Y shape: ", zoom_data_Y.shape)
                
                sliceUsed(dataA=zoom_data_X, dataB=zoom_data_Y, chanA=input_chan, chanB=output_chan,
                          name_dataset=name_dataset, name_tag=filename_Y, resize_f=1)
                
            print("------------------------------------------------------------------------")

if __name__ == "__main__":
    main()