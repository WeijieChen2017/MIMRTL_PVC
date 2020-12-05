import os
import glob
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import xlwt
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity

def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--nameDataset', metavar='', type=str, default="sk8R",
                        help='Name for the dataset needed to be reverse.(sk8R)<str>')
    parser.add_argument('--dataFolder', metavar='', type=str, default="pMix",
                        help='Folder of testing dataset.(unet)<str>')

    args = parser.parse_args()
    name_dataset = args.nameDataset
    data_folder = args.dataFolder

    print("name_dataset=", name_dataset)
    print("data_folder=", data_folder)

    eval_results(name_dataset=name_dataset, data_folder=data_folder)


def eval_results(name_dataset="sk8R", data_folder="pMix"):
    recon_list = glob.glob("./data/"+data_folder+"/recon/*.nii")
    recon_list.sort()
    metric_xls = xlwt.Workbook()

    metric_tab_eval = metric_xls.add_sheet("eval")
    metric_tab_eval.write(0, 1, "NRMSE")
    metric_tab_eval.write(0, 2, "PSNR")
    metric_tab_eval.write(0, 3, "SSIM")

    for idx_p, recon_path in enumerate(recon_list):
        recon_name = os.path.basename(recon_path)
        gt_name = recon_name[:-len(name_dataset)-13]+"rec.nii"
        gt_path = "./data/"+data_folder+"/gt/"+gt_name
        metric_tab_eval = metric_xls.add_sheet(gt_name[17:20])
        metric_tab_eval.write(idx_p+1, 0, gt_name[17:20])
        
        print("-------------------------------------")
        print(gt_name)
        
        recon_nii = nib.load(recon_path)
        gt_nii = nib.load(gt_path)
        
        recon_data = recon_nii.get_fdata()
        gt_data = gt_nii.get_fdata()
        
        dx, dy, dz = recon_data.shape
        
        metric_methods_hub = [normalized_root_mse, peak_signal_noise_ratio, structural_similarity]
    #     metric_table = np.zeros((len(metric_methods_hub), dz))
        metric_tab.write(0, 1, "NRMSE")
        metric_tab.write(0, 2, "PSNR")
        metric_tab.write(0, 3, "SSIM")
        
        for idx_z in range(dz):
            metric_tab.write(idx_z+1, 0, idx_z+1)
        for idx_m, metric_method in enumerate(metric_methods_hub):
            qualified_metric = []
            for idx_z in range(dz):
                im_true = gt_data[:, :, idx_z]
                im_test = recon_data[:, :, idx_z]
    #             metric_table[idx_m, idx_z] = metric_method(im_true, im_test)
                if np.mean(im_true) > 1e-3:
                    current_metric = metric_method(im_true, im_test)
                    qualified_metric.append(current_metric)
                    metric_tab.write(idx_z+1, idx_m+1, current_metric)
            qualified_metric = np.asarray(qualified_metric)
            metric_tab_eval.write(idx_p+1, idx_m+1, np.mean(qualified_metric))
    metric_xls.save("M-"+name_dataset+"+D-"+data_folder+".xls")
    print("------------------------------------------------------------")
    print("----------------------Finished------------------------------")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()