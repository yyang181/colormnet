import sys
import os
import cv2
import numpy as np

import math
from PIL import Image
from datetime import datetime
from skimage.metrics import structural_similarity
import lpips
import torch

from fid import calculate_fid, calculate_fid_yyx
from cdc import calculate_cdc, calculate_cdc_yyx 



# D. Hasler and S. E. Suesstrunk, “Measuring colorfulness in natural images,” in Human Vision and Electronic Imaging VIII, B. E. Rogowitz and T. N. Pappas, Eds., vol. 5007, International Society for Optics and Photonics. SPIE, 2003, pp. 87 – 95. [Online]. Available: https://doi.org/10.1117/12.477378 7
def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

def calculate_psnr(img1, img2):
    mse_value = ((img1 - img2)**2).mean()
    if mse_value == 0:
        result = float('inf')
    else:
        result = 20. * np.log10(255. / np.sqrt(mse_value))
    return result

def calculate_psnr_for_folder(gt_folder, result_folder, loss_fn_alex, flag_CalPSNR, flag_CalSSIM, flag_CalLPIPS, flag_CalColorfulness):
    result_clips = sorted([clip for clip in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, clip))])

    psnr_values = []
    ssim_values = []
    colorfulness = []
    lpips = []
    for clip in result_clips:
        path_clip = os.path.join(result_folder, clip)
        test_files = sorted([img for img in os.listdir(path_clip) if img.endswith('.png') or img.endswith('.jpg')])

        for img in test_files:
            gt_path = os.path.join(gt_folder, clip, img)
            result_path = os.path.join(path_clip, img)
            
            gt_img = np.array(Image.open(gt_path).convert('RGB'))
            result_img = np.array(Image.open(result_path).convert('RGB'))

            if flag_CalPSNR:
                psnr = calculate_psnr(gt_img, result_img)
            else:
                psnr = 0.0

            if flag_CalSSIM:    
                ssim_score = structural_similarity(gt_img, result_img, channel_axis=2)
            else:
                ssim_score = 0.0

            if flag_CalColorfulness:
                colorfulness_score = image_colorfulness(result_img)
            else:
                colorfulness_score = 0.0

            if flag_CalLPIPS:
                # lpips
                torcha = torch.unsqueeze(torch.from_numpy(np.transpose(gt_img, (2,0,1))), 0)
                torchb = torch.unsqueeze(torch.from_numpy(np.transpose(result_img, (2,0,1))), 0)
                lpips_score = loss_fn_alex(torcha, torchb)[0,0,0,0].item()
            else:
                lpips_score = 0.0
            
            psnr_values.append(psnr)
            ssim_values.append(ssim_score)
            colorfulness.append(colorfulness_score)
            lpips.append(lpips_score)
        
        # print(clip, psnr, ssim_score, colorfulness_score, lpips_score)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_colorfulness = np.mean(colorfulness)
    avg_lpips = np.mean(lpips)
    return avg_psnr, avg_ssim, avg_colorfulness, avg_lpips

flag_CalFid = False
flag_CalCDC = False
flag_CalPSNR = True # True False
flag_CalSSIM = True
flag_CalLPIPS = False
flag_CalColorfulness = False

# Input folder only for an example here, please change the path to your own folder
GT_dataset_folder = './input/'
real_save_folder = './input/'

if flag_CalFid:
    print('Calculating FID for %s'%(real_save_folder))
    fid = calculate_fid_yyx(real_save_folder, GT_dataset_folder, 100)
    print('%s FID: %s'%(real_save_folder, fid))

if flag_CalCDC:
    print('Calculating CDC for %s'%(real_save_folder))
    cdc = calculate_cdc_yyx(real_save_folder)
    print('%s CDC: %s'%(real_save_folder, cdc))

if flag_CalPSNR or flag_CalSSIM or flag_CalLPIPS or flag_CalColorfulness:
    if flag_CalLPIPS:
        loss_fn_alex = lpips.LPIPS(net='alex', verbose=False) # best forward scores
    else:
        loss_fn_alex = None
    print('Calculating PSNR SSIM lpips colorfulness for %s'%(real_save_folder))
    
    psnr, ssim, colorfulness, lpips_score = calculate_psnr_for_folder(GT_dataset_folder, real_save_folder, loss_fn_alex, flag_CalPSNR, flag_CalSSIM, flag_CalLPIPS, flag_CalColorfulness)
    print('%s PSNR: %s SSIM %s lpips %s colorfulness %s'%(real_save_folder, psnr, ssim, lpips_score, colorfulness))

# if flag_CalPSNR:
#     print('Calculating PSNR for %s %s'%(dataset, method_name))
#     psnr = calculate_psnr_for_folder(GT_dataset_folder, real_save_folder)
#     print('%s %s PSNR: %s'%(dataset, method_name, cdc))

if flag_CalPSNR and flag_CalCDC and flag_CalFid:
    print('*********************************************************************************************************')
    print('%s FID: %s CDC: %s PSNR: %s SSIM %s lpips %s colorfulness %s'%(real_save_folder, fid, cdc, psnr, ssim, lpips_score, colorfulness))




