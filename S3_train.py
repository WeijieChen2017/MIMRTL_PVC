import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--nameDataset', metavar='', type=str, default="sk8R",
                        help='Name for the dataset needed to be reverse.(sk8R)<str>')

    args = parser.parse_args()
    name_dataset = args.nameDataset

    command = "python train.py"
    command += " --dataroot ./datasets/"+name_dataset
    command += " --name "+name_dataset
    command += " --model "+"pix2pix"
    command += " --batch_size "+"16"
    command += " --gpu_ids "+"0"
    command += " --save_epoch_freq "+"50"
    command += " --n_epochs "+"100"
    command += " --n_epochs_decay "+"100"
    command += " --input_nc "+"7"
    command += " --output_nc "+"7"
    command += " --netG "+"unet_512"
    command += " --direction "+"AtoB"
    command += " --norm "+"batch"

    print(command)
    os.chdir("pytorch-CycleGAN-and-pix2pix")
    os.system(command)


if __name__ == "__main__":
    main()