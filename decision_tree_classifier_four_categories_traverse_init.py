"""
move all used validation resources (images,ground truth, etc.) to corresponding CRF-related folder.
This is a prerequisite step if you want to run test for CRF.
"""

import sys
import random
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cPickle
import copy
from shutil import copy

if __name__ == '__main__':
    dataset='val'

    original_image_folder = '/home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'+dataset+'/'
    original_image_files_all=glob.glob(os.path.join(original_image_folder,"*","*.png"))
    original_image_files_all.sort()

    original_image_folder_for_traverse='/home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'+dataset+'_for_traverse/'
    if not os.path.exists(original_image_folder_for_traverse):
        os.makedirs(original_image_folder_for_traverse)


    gt_folder = '/home/panquwang/Dataset/CityScapes/gtFine/'+dataset+'/'
    gt_color_files_all=glob.glob(os.path.join(gt_folder,"*","*gtFine_color.png"))
    gt_color_files_all.sort()

    gt_label_images_all = glob.glob(os.path.join(gt_folder,"*","*labelIds.png"))
    gt_label_images_all.sort()

    gt_jsons_all = glob.glob(os.path.join(gt_folder,"*","*polygons.json"))
    gt_jsons_all.sort()

    gt_instanceids_all = glob.glob(os.path.join(gt_folder,"*","*instanceIds.png"))
    gt_instanceids_all.sort()

    gt_dir_for_traverse = '/home/panquwang/Dataset/CityScapes/gtFine/'+dataset+'_for_traverse/'
    if not os.path.exists(gt_dir_for_traverse):
        os.makedirs(gt_dir_for_traverse)


    superpixel_result_folder='/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_'+dataset+'/data/'
    superpixel_data_all=glob.glob(os.path.join(superpixel_result_folder,'*.dat'))
    superpixel_data_all.sort()

    superpixel_result_folder_for_traverse='/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_'+dataset+'_for_traverse/data/'
    if not os.path.exists(superpixel_result_folder_for_traverse):
        os.makedirs(superpixel_result_folder_for_traverse)




    # prediction for validation set
    folder={}
    # base
    folder[1]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/',dataset, dataset+'-epoch-35-CRF', 'score')
    # truck, wall
    folder[2]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/',dataset, dataset+'-epoch-39-CRF-050', 'score')
    # bus, train
    folder[3]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_atrous16_epoch_33/', dataset, dataset+'-epoch-33-CRF', 'score')

    folder_files={}
    for key,value in folder.iteritems():
        folder_files[key]=glob.glob(os.path.join(value,'*.png'))
        folder_files[key].sort()

    folder_1_label_map_for_traverse=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/',dataset, dataset+'-epoch-35-CRF_for_traverse', 'score')
    if not os.path.exists(folder_1_label_map_for_traverse):
        os.makedirs(folder_1_label_map_for_traverse)
    folder_2_label_map_for_traverse=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/',dataset, dataset+'-epoch-39-CRF-050_for_traverse', 'score')
    if not os.path.exists(folder_2_label_map_for_traverse):
        os.makedirs(folder_2_label_map_for_traverse)
    folder_3_label_map_for_traverse=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_atrous16_epoch_33/', dataset, dataset+'-epoch-33-CRF_for_traverse', 'score')
    if not os.path.exists(folder_3_label_map_for_traverse):
        os.makedirs(folder_3_label_map_for_traverse)

    random_list=[   1,   2,   3,   4,   7,   8,   14,  17,  21,  24,  26,  28,  32,
                    34,  35,  42,  43,  46,  48,  50,  53,  54,  56,  58,  61,  64,
                    65,  66,  75,  76,  77,  78,  82,  85,  86,  92,  93, 101, 102,
                   106, 111, 112, 113, 114, 115, 116, 118, 120, 121, 127, 131, 134,
                   136, 138, 140, 152, 154, 156, 162, 167, 170, 174, 178, 181, 183,
                   184, 186, 188, 192, 198, 199, 201, 203, 204, 205, 206, 207, 211,
                   213, 215, 218, 223, 225, 226, 228, 229, 234, 235, 238, 251, 253,
                   259, 261, 263, 267, 271, 273, 275, 280, 284, 292, 299, 301, 315,
                   316, 317, 319, 322, 326, 328, 330, 334, 335, 337, 338, 351, 353,
                   357, 358, 360, 362, 368, 382, 387, 392, 396, 398, 403, 411, 418,
                   420, 421, 424, 435, 437, 438, 443, 445, 454, 459, 472, 478, 479,
                   482, 484, 487, 488, 494, 495, 498]

    original_image_files  =  [original_image_files_all[i] for i in random_list]
    superpixel_data = [superpixel_data_all[i] for i in random_list]
    gt_color_files = [gt_color_files_all[i] for i in random_list]
    gt_label_images = [gt_label_images_all[i] for i in random_list]
    gt_jsons = [gt_jsons_all[i] for i in random_list]
    gt_instanceids = [gt_instanceids_all[i] for i in random_list]
    folder_1_label_maps = [folder_files[1][i] for i in random_list]
    folder_2_label_maps = [folder_files[2][i] for i in random_list]
    folder_3_label_maps = [folder_files[3][i] for i in random_list]

    for i in range(len(random_list)):
        print i
        copy(original_image_files[i], original_image_folder_for_traverse)
        copy(superpixel_data[i], superpixel_result_folder_for_traverse)
        copy(gt_color_files[i], gt_dir_for_traverse)
        copy(gt_label_images[i], gt_dir_for_traverse)
        copy(gt_jsons[i], gt_dir_for_traverse)
        copy(gt_instanceids[i], gt_dir_for_traverse)
        copy(folder_1_label_maps[i], folder_1_label_map_for_traverse)
        copy(folder_2_label_maps[i], folder_2_label_map_for_traverse)
        copy(folder_3_label_maps[i], folder_3_label_map_for_traverse)