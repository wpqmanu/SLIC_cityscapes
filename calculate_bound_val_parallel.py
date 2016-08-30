import cv2
import sys
import scipy
import scipy.linalg
import random
import math
import os.path
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import glob
import cPickle
from datetime import datetime
import collections
sys.path.append( os.path.normpath( os.path.join('/home/panquwang/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
import labels
from labels     import trainId2label,id2label
from collections import Counter
from PIL import Image
import copy
from joblib import Parallel, delayed
import multiprocessing



def convert_label_to_trainid(current_layer_value):
    # convert label
    unique_values_in_array = np.unique(current_layer_value)
    unique_values_in_array = np.sort(unique_values_in_array)
    for unique_value in unique_values_in_array:
        converted_value = id2label[unique_value].trainId
        current_layer_value[current_layer_value == unique_value] = converted_value
    return  current_layer_value

def convert_trainid_to_label(label):
    unique_values_in_final_array = np.unique(label)
    unique_values_in_final_array = np.sort(unique_values_in_final_array)
    unique_values_in_final_array = unique_values_in_final_array[::-1]
    for unique_value in unique_values_in_final_array:
        if unique_value < 19:
            converted_value = trainId2label[unique_value].id
        else:
            converted_value = 255
        label[label == unique_value] = converted_value
    label[label == 255] = 0
    return label


def get_palette():
    # get palette
    trainId2colors = {label.trainId: label.color for label in labels.labels}
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette

def calculate_pixelwise_upper_bound(gt_files,folder_files):
    result_location='/mnt/scratch/panqu/SLIC/bounds/pixelwse_upper_bound_results/'
    if not os.path.exists(result_location):
        os.makedirs(result_location)

    # calculation of upper bound
    # iterate through all images
    for index in range(len(gt_files)):
        print index
        current_gt=cv2.imread(gt_files[index],0)
        file_name=gt_files[index].split('/')[-1]
        # gather prediction maps
        total_area=1024*2048
        current_all_layer_values=np.zeros((1024,2048,len(folder_files)))
        for key, value in folder.iteritems():
            current_layer_value = cv2.imread(folder_files[key][index], 0)
            # convert label
            current_all_layer_values[:,:,key-1] = current_layer_value

        near_perfect_map=np.zeros((1024,2048))
        for row_id in range(near_perfect_map.shape[0]):
            for col_id in range(near_perfect_map.shape[1]):
                if current_gt[row_id][col_id] in current_all_layer_values[row_id][col_id]:
                    near_perfect_map[row_id][col_id]=current_gt[row_id][col_id]

        cv2.imwrite(os.path.join(result_location,file_name),near_perfect_map)


def parallel_processing(index, original_image_files,gt_files,gt_files_color,folder_files,superpixel_data):
    result_location='/mnt/scratch/panqu/SLIC/bounds/superpixelwse_upper_bound_results/'
    if not os.path.exists(result_location):
        os.makedirs(result_location)
        os.makedirs(os.path.join(result_location,'score'))
        os.makedirs(os.path.join(result_location,'visualization'))

    # calculation of upper bound
    # iterate through all images
    print index
    file_name=gt_files[index].split('/')[-1]
    original_image=cv2.imread(original_image_files[index])


    # gather prediction maps, form multi-layer maps
    current_all_layer_values = np.zeros((1024, 2048, len(folder_files)))
    for key, value in folder.iteritems():
        current_layer_value = cv2.imread(folder_files[key][index], 0)
        current_all_layer_values[:, :, key - 1]=convert_label_to_trainid(current_layer_value)

    # gather superpixel data and ground truth
    current_superpixel_data = cPickle.load(open(superpixel_data[index], "rb"))
    current_gt=cv2.imread(gt_files[index],0)

    superpixel_label = current_superpixel_data[2]

    superpixel_set=np.unique(superpixel_label)
    num_superpixels = len(superpixel_set)

    # iterate through all superpixels
    near_perfect_map=np.ones((1024,2048))*255
    for index_superpixel in superpixel_set:
        gt_label=current_gt[superpixel_label == index_superpixel]
        gt_label=gt_label[gt_label!=255]
        gt_label_count=Counter(gt_label).most_common()
        # gt_label_consistency_rate = float(gt_label_count[0][1]) / len(gt_label)
        if len(gt_label_count)>0:
            gt_label_candidates=[a[0] for a in gt_label_count]
            gt_label_selected=gt_label_count[0][0]
        else: # the current superpixel region only has ignore label
            continue

        for layer_index in range(current_all_layer_values.shape[2]):
            current_layer_current_superpixel_label = current_all_layer_values[:,:,layer_index][superpixel_label == index_superpixel]
            current_layer_label_count = Counter(current_layer_current_superpixel_label).most_common()
            current_layer_labels=[a[0] for a in current_layer_label_count]
            current_layer_consistency_rate = float(current_layer_label_count[0][1]) / len(current_layer_current_superpixel_label)
            # if by any chance we encounter gt_label, we assign gt_label to current superpixel
            if gt_label_selected in current_layer_labels:
                near_perfect_map[superpixel_label == index_superpixel]=gt_label_selected
                break

    # save score
    near_perfect_map_saved=copy.deepcopy(near_perfect_map)
    score=convert_trainid_to_label(near_perfect_map)
    cv2.imwrite(os.path.join(result_location,'score',file_name),score)

    # save visualization
    # original image
    concat_img = Image.new('RGB', (2048*3, 1024))
    concat_img.paste(Image.fromarray(original_image[:, :, [2, 1, 0]]).convert('RGB'), (0, 0))
    # ground truth
    gt_img = Image.open(gt_files_color[index])
    concat_img.paste(gt_img, (2048, 0))
    # prediction
    near_perfect_map_saved=near_perfect_map_saved.astype(np.uint8)
    result_img = Image.fromarray(near_perfect_map_saved).convert('P')
    palette=get_palette()
    result_img.putpalette(palette)
    # concat_img.paste(result_img, (2048*2,0))
    concat_img.paste(result_img.convert('RGB'), (2048*2,0))
    concat_img.save(os.path.join(result_location, 'visualization', file_name))
#
# def calculate_superpixelwise_lower_bound(gt_files,folder_files):

if __name__ == '__main__':
    gt_folder='/home/panquwang/Dataset/CityScapes/gtFine/val/'
    gt_files=glob.glob(os.path.join(gt_folder,"*","*gtFine_labelTrainIds.png"))
    gt_files.sort()
    gt_files_color=glob.glob(os.path.join(gt_folder,"*","*gtFine_color.png"))
    gt_files_color.sort()

    original_image_folder='/home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/val/'
    original_image_files=glob.glob(os.path.join(original_image_folder,"*","*.png"))
    original_image_files.sort()

    # base feature map
    folder={}
    # base
    folder[1]='/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/val/val-epoch-35-CRF/score/'
    # truck, wall
    folder[2]='/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/val/val-epoch-39-CRF-050/score/'
    # bus, train
    folder[3]='/mnt/scratch/panqu/to_pengfei/asppp_atrous16_epoch_33/val/val-epoch-33-CRF/score/'
    folder_files={}
    previous_key=0
    for key,value in folder.iteritems():
        folder_files[key]=glob.glob(os.path.join(value,'*.png'))
        folder_files[key].sort()


    superpixel_folder='/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_val/data/'
    superpixel_files=glob.glob(os.path.join(superpixel_folder,'*.dat'))
    superpixel_files.sort()

    # lower bound: majority vote. Upperbound: correct if appears.
    # get label map for pixelwise upper bound. Uncomment if not necessary
    # calculate_pixelwise_upper_bound(gt_files,folder_files)


    # get "upper" bound for superpixel implementation.Uncomment if not necessary
    num_cores = multiprocessing.cpu_count()
    range_i=range(0,500)
    Parallel(n_jobs=num_cores)(delayed(parallel_processing)(i,original_image_files,gt_files,gt_files_color,folder_files,superpixel_files) for i in range_i)



    # get "upper" bound for superpixel implementation.Uncomment if not necessary
    # calculate_superpixelwise_upper_bound(index,original_image_files,gt_files,gt_files_color,folder_files,superpixel_files)
    #
