import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--suffixes', metavar='', type=str, default="vanilla",
                        help='Suffixes for the current model.(vanilla)<str>')
    parser.add_argument('--lr', metavar='', type=float, default=0.0002,
                        help='Learning rate(0.0002)<float>')




    parser.add_argument('--dir_pet', metavar='', type=str, default="breast1_pet",
                        help='Name of PET subject.(breast1_pet)<str>')
    parser.add_argument('--dir_mri', metavar='', type=str, default="breast1_water",
                        help='Name of MRI subject.(breast1_water)<str>')
    parser.add_argument('--blur_method', metavar='', type=str, default="nib_smooth",
                        help='The process method of synthesizing PET(nib_smooth)<str> [kernel_conv/skimage_gaus/nib_smooth]')
    parser.add_argument('--blur_para', metavar='', type=str, default="4",
                        help='Parameters of process data(4)<str>')
    parser.add_argument('--slice_x', metavar='', type=int, default="1",
                        help='Slices of input(1)<int>[1/3]')
    parser.add_argument('--enhance_blur', metavar='', type=bool, default=True,
                        help='Whether stack different process methods to train the model')
    parser.add_argument('--id', metavar='', type=str, default="eeVee",
                        help='ID of the current model.(eeVee)<str>')


    parser.add_argument('--epoch', metavar='', type=int, default=500,
                        help='Number of epoches of training(300)<int>')
    parser.add_argument('--n_filter', metavar='', type=int, default=64,
                        help='The initial filter number(64)<int>')
    parser.add_argument('--depth', metavar='', type=int, default=4,
                        help='The depth of U-Net(4)<int>')
    parser.add_argument('--batch_size', metavar='', type=int, default=10,
                        help='The batch_size of training(10)<int>')
    parser.add_argument('--dlink', metavar='', type=int, default=0,
                        help='The number of dilation blocks (0)<int>')

    parser.add_argument('--noise_flag', metavar='', type=bool, default=False,
                        help='Do you want to add guassian noise? (True)<bool>')
    parser.add_argument('--noise_mean', metavar='', type=float, default=0,
                        help='The mean of gaussian noise if added (0)<float>')
    parser.add_argument('--noise_sigma', metavar='', type=float, default=1e-3,
                        help='The variance of gaussian noise if added (1e-3)<float>')


    # para box
    parser.add_argument('--flag_grid_search', metavar='', type=bool, default=False,
                        help='whether to start the grid search')
    parser.add_argument('--fwhm', metavar='', type=float, default=float,
                        help='FWHM')
    parser.add_argument('--gau_sigma', metavar='', type=float, default=0,
                        help='The relative sigma of gaussian noise')
    parser.add_argument('--poi_sigma', metavar='', type=float, default=0,
                        help='The relative sigma of poisson noise')



    args = parser.parse_args()


    # model_name = args.model_name
    # dataset_folder_name = args.dataset_folder_name

    # dir_mri = os.path.join("data", args.dir_mri + '.nii')
    # dir_pet = os.path.join("data", args.dir_pet + '.nii')

    # time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    # model_id = args.id + time_stamp
    # enhance_blur = args.enhance_blur
    # gbl_set_value("depth", args.depth)

    command_0 = "pytorch-CycleGAN-and-pix2pix"
    print("------------------------")
    print(command_0)
    os.chdir(command_0)

    command_1 = "cp checkpoints/sk8R_" + args.suffixes + "/10_net_G_B.pth " + "checkpoints/sk8R_" + args.suffixes + "/latest_net_G.pth"
    print("------------------------")
    print(command_1)
    os.system(command_1)

    command_2 = "python test.py --dataroot ./datasets/sk8R --model test --name sk8R_vanilla --dataset_mode single --num_test 300 --input_nc 7 --output_nc 7 --no_dropout --netG unet_512 --norm instance"
    print("------------------------")
    print(command_2)
    os.system(command_2)

    command_3 = ".."
    print("------------------------")
    print(command_3)
    os.chdir(command_3)

    command_4 = "python S5_Assembler.py"
    print("------------------------")
    print(command_4)
    os.system(command_4)

    command_5 = "cp ./subj01_pet_sk8R_" + args.suffixes + "_fake.nii ~/" + args.suffixes + ".nii"
    print("------------------------")
    print(command_5)
    os.system(command_5)


if __name__ == "__main__":
    main()