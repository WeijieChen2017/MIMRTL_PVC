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

    parser.add_argument('--showOnly', metavar='', type=str, default="True",
                        help='Whether to only show commands instead of excuting them.(True)<str>')

    args = parser.parse_args()
    print(args)
    show_only = args.showOnly

    for subj in [1,2,3]:
        for pf in [1,2,3,5,7]:#, "p2", "p3", "pMix"

            name = "subj0"+str(subj)+"_pet_k8p"+str(pf)+"b8_wb_fake"

            command_0 = "sshpass -p \"pokemon151\" scp -r \"wchen@cn0.medphysics.wisc.edu:/data/users/wchen/github/MIMRTL_PVC/" + name
            command_0 += ".nii\" \"/Volumes/NOOK/WIMR/report/wholeBody/"+name
            command_0 += ".nii\""
            print(command_0)
        
        # print(command_6)

            print("Show only? ",show_only)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

            if show_only != "True":
                os.system(command_0)
                # os.system(command_1)
                # os.system(command_2)
                # os.system(command_3)
                # os.system(command_4)
                # os.system(command_5)
                # os.system(command_6)

        
    # command_0 = "pytorch-CycleGAN-and-pix2pix"
    # print("------------------------")
    # print(command_0)
    # os.chdir(command_0)

    # command_1 = "cp checkpoints/sk8R_" + args.suffixes + "/10_net_G_B.pth " + "checkpoints/sk8R_" + args.suffixes + "/latest_net_G.pth"
    # print("------------------------")
    # print(command_1)
    # os.system(command_1)

    # command_2 = "python test.py --dataroot ./datasets/sk8R --model test --name sk8R_vanilla --dataset_mode single --num_test 300 --input_nc 7 --output_nc 7 --no_dropout --netG unet_512 --norm instance"
    # print("------------------------")
    # print(command_2)
    # os.system(command_2)

    # command_3 = ".."
    # print("------------------------")
    # print(command_3)
    # os.chdir(command_3)

    # command_4 = "python S5_Assembler.py"
    # print("------------------------")
    # print(command_4)
    # os.system(command_4)

    # command_5 = "cp ./subj01_pet_sk8R_" + args.suffixes + "_fake.nii ~/" + args.suffixes + ".nii"
    # print("------------------------")
    # print(command_5)
    # os.system(command_5)

    # os.system(command_0)


    # command_1 = "python S1_simulation.py --nameDataset " + name
    # command_1 += " --fwhmHub "+str(fwhm)
    # command_1 += " --gauHub 0 --poiHub 0"
    # # print(command_1)
    # # os.system(command_1)

    # command_2 = "python S2_createDataset.py --nameDataset "+name
    # command_2 += " --outputChannel 1"
    # # print(command_2)
    # # os.system(command_2)

    # command_3 = "python S3_train.py --nameDataset "+name
    # # print(command_3)
    # # os.system(command_3)

    # command_4 = "python S4_test.py --nameDataset "+name
    # # print(command_4)
    # # os.system(command_4)

    # command_5 = "python S5_assembler.py --nameDataset "+name
    # # print(command_5)
    # # os.system(command_5)

    # command_6 = "rm -r ./pytorch-CycleGAN-and-pix2pix/*/"+name

    # unet_256
    # python S2_createDataset.py --nameDataset BraTS_1024 --inputChannel 5 --outputChannel 1
    # python train.py --dataroot ./datasets/Gibbs --name Gibbs_m3 --model pix2pix --batch_size 80 --gpu_ids 0 --save_epoch_freq 50 --n_epochs 200 --n_epochs_decay 200 --input_nc 1 --output_nc 1 --netG resnet_6blocks --direction AtoB --norm batch
    # python test.py --dataroot ./datasets/Gibbs --name Gibbs_m3 --model test --num_test 1780 --dataset_mode single --input_nc 1 --output_nc 1 --netG resnet_6blocks --direction AtoB --norm batch --no_dropout
    # python S5_assembler.py --nameDataset BraTS_DA5_R9 --dataFolder BraTS
    # python S8_eval.py --nameDataset BraTS_DA5_R9 --dataFolder BraTS
    # rm -r ./pytorch-CycleGAN-and-pix2pix/*/oct19_9
    command += " --name "+name_dataset
    command += " --model "+"test"
    command += " --num_test "+"900"
    command += " --dataset_mode "+"single"
    command += " --input_nc "+"7"
    command += " --output_nc "+"1"
    command += " --netG "+"unet_512"
    command += " --direction "+"AtoB"
    command += " --norm batch"
    command += " --no_dropout"


if __name__ == "__main__":
    main()