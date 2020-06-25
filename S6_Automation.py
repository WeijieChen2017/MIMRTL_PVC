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

    for powerFactor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:#, "p2", "p3", "pMix"

        if powerFactor <1:
            name = "k8p"+str(int(powerFactor*10))+"v"
        else:
            name = "k8p"+str(powerFactor)+"v"

        command_0 = "python S0_reverse.py --nameDataset " + name
        command_0 += " --powerFactor "+str(powerFactor)
        print(command_0)
        # os.system(command_0)

        command_1 = "python S1_simulation.py --nameDataset " + name
        command_1 += " --fwhmHub 8"
        command_1 += " --gauHub 0 --poiHub 0"
        print(command_1)
        # os.system(command_1)

        command_2 = "python S2_createDataset.py --nameDataset "+name
        command_2 += " --outputChannel 1"
        print(command_2)
        # os.system(command_2)

        command_3 = "python S3_train.py --nameDataset "+name
        print(command_3)
        # os.system(command_3)

        command_4 = "python S4_test.py --nameDataset "+name
        print(command_4)
        # os.system(command_4)

        command_5 = "python S5_assembler.py --nameDataset "+name
        print(command_5)
        # os.system(command_5)

        command_6 = "rm -r ./pytorch-CycleGAN-and-pix2pix/*/"+name
        print(command_6)

        command_7 = "rm -r ./data/"+name
        print(command_7)

        print("Show only? ",show_only)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        if show_only != "True":
            os.system(command_0)
            os.system(command_1)
            os.system(command_2)
            os.system(command_3)
            os.system(command_4)
            os.system(command_5)
            os.system(command_6)
            os.system(command_7)

        
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


if __name__ == "__main__":
    main()