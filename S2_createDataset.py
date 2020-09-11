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



    args = parser.parse_args()
    name_dataset = args.nameDataset
    name_model = args.nameModel
    input_chan = args.inputChannel
    output_chan = args.outputChannel
    resize_f = args.resizeFactor
    save_flag = args.save
    norm = args.norm

    print("name_dataset=", name_dataset)
    print("name_model=", name_model)
    print("input_chan=", input_chan)
    print("output_chan=", output_chan)
    print("resize_f=", resize_f)
    print("normalization=", norm)
    print("save_flag=", save_flag)

    create_dataset(name_dataset=name_dataset, name_model = name_model,
                   input_chan=input_chan, output_chan=output_chan, resize_f=resize_f,
                   norm=norm, save_flag=save_flag)



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
    # emptySlice = np.zeros((h,w,1))
    # dataA = np.concatenate((dataA, emptySlice), axis=2)
    print(dataA.shape)
        
    for idx_z in range(z):
        for idx_c in range(n_slice):
            img[idx_c, :, :] = zoom(dataA[:, :, int(index[idx_z, idx_c])], zoom=resize_f)
        name2save = path2save+name_tag+"_"+str(idx_z)+".npy"
        np.save(name2save, img)
    print(str(z)+" images have been saved.")

def data_aug(imgA, imgB):
    
    shear_hub = []
    shift_hub = []
    scale_hub = []
    rot_hub = []
    crop_hub = []
    mirrot_hub = []


    return imgA, imgB


def createAB(dataA, dataB, name_dataset, chanA=7, chanB=1,
             name_tag="", resize_f=1, folderName='test', save_flag=False):
    # shape supposed to be 512*512*284 by default
    assert dataA.shape == dataB.shape, ("DataA should share the same shape with DataB.")
    path2save = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/train/"
    h, w, z = dataA.shape
    h = h*resize_f
    w = w*resize_f
    
    imgA = np.zeros((chanA, h, w))
    imgB = np.zeros((chanB, h, w))
    indexA = create_index(dataA, chanA)
    indexB = create_index(dataB, chanB)
    # print(np.amax(dataB), np.amin(dataB), dataB.shape )

    # print(indexA)
    # print(indexB)
    for idx_z in range(z):

        for idx_a in range(chanA):
            imgA[idx_a, :, :] = zoom(dataA[:, :, int(indexA[idx_z, idx_a])], zoom=resize_f)
        for idx_b in range(chanB):
            imgB[idx_b, :, :] = zoom(dataB[:, :, int(indexB[idx_z, idx_b])], zoom=resize_f)

        # imgA, imgB = data_aug(imgA, imgB)
        # print(np.amax(imgA), np.amin(imgA))
        # print(np.amax(imgB), np.amin(imgB))
        imgA = maxmin_norm(imgA)
        imgB = maxmin_norm(imgB)

        # print(imgB)
        # print(np.amax(imgA), np.amin(imgA))
        # print(np.amax(imgB), np.amin(imgB))

        img = [imgA, imgB]

        name2save = path2save+name_tag+"_"+str(idx_z)+".npy"
        # if save_flag:
        np.save(name2save, img)
    print(str(z)+" images have been saved.")


def create_dataset(name_dataset='sk8R', name_model = "unet", input_chan=7, output_chan=7,
                   resize_f=1, norm="maxmin", save_flag=False):

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

    list_ori = glob.glob("./data/"+name_dataset+"/test/*.nii")
    list_ori.sort()
    print("Test:")
    for path_ori in list_ori:
        filename_ori = os.path.basename(path_ori)[:]
        filename_ori = filename_ori[:filename_ori.find(".")]
        print(filename_ori)
        data_ori = normUsed(nib.load(path_ori).get_fdata())
        slice5_A(dataA=data_ori, name_dataset=name_dataset, n_slice=input_chan,
                  name_tag=filename_ori, resize_f = resize_f, folderName='test')
        print("------------------------------------------------------------------------")

    # list_ori = glob.glob("./data/"+name_dataset+"/pure/*.nii")
    # list_ori.sort()
    # print("Pure:")
    # for path_ori in list_ori:
    #     filename_ori = os.path.basename(path_ori)[:]
    #     filename_ori = filename_ori[:filename_ori.find(".")]
    #     print(filename_ori)
    #     data_ori = normUsed(nib.load(path_ori).get_fdata())
    #     slice5_A(dataA=data_ori, name_dataset=name_dataset, n_slice=input_chan, 
    #              name_tag=filename_ori, resize_f = resize_f, folderName='trainA')
    #     print("------------------------------------------------------------------------")

    # list_ori = glob.glob("./data/"+name_dataset+"/blur/*.nii")
    # list_ori.sort()
    # print("Blur:")
    # for path_ori in list_ori:
    #     filename_ori = os.path.basename(path_ori)[:]
    #     filename_ori = filename_ori[:filename_ori.find(".")]
    #     print(filename_ori)
    #     data_ori = normUsed(nib.load(path_ori).get_fdata())
    #     slice5_A(dataA=data_ori, name_dataset=name_dataset, n_slice=output_chan,
    #              name_tag=filename_ori, resize_f = resize_f, folderName='trainB')
    #     print("------------------------------------------------------------------------")




    list_ori = glob.glob("./data/"+name_dataset+"/pure/*.nii")
    list_ori.sort()
    for path_ori in list_ori:
        print("TrainA:")
        filename_ori = os.path.basename(path_ori)[:]
        filename_ori = filename_ori[:filename_ori.find(".")]
        print(filename_ori)
        data_ori = normUsed(nib.load(path_ori).get_fdata())
        
        list_sim = glob.glob("./data/"+name_dataset+"/blur/*"+filename_ori+"*.nii")
        list_sim.sort()
        
        for path_sim in list_sim:
            print("Pairs")
            filename_sim = os.path.basename(path_sim)[:]
            filename_sim = filename_sim[:filename_sim.rfind(".")]
            print("A:", filename_sim)
            print("B:", filename_ori)
                    
            data_sim = normUsed(nib.load(path_sim).get_fdata())
            sliceUsed(dataA=data_sim, dataB=data_ori, chanA=7, chanB=1,
                      name_dataset=name_dataset, name_tag=filename_sim, resize_f=1)
            
        print("------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
        
        
#         data_ori = nib.load(path_ori).get_fdata()
#         norm_ori = maxmin_norm(data_ori)*255
#         sliced_save(data=norm_ori,
#                     name_tag=os.path.basename(path_ori)[:-4],
#                     path2save="./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/"+folder_name+"/",
#                     n_slice=n_slice)