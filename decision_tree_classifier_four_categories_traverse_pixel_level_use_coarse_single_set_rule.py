from sklearn.ensemble import RandomForestClassifier
import sys
import scipy
import scipy.linalg
import random
import math
import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import cPickle
from PIL import Image
from datetime import datetime
import collections
import copy
from collections import Counter
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import get_feature_single_superpixel
import feature_extraction_with_neighbor
# sys.path.insert(0,'/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
import cv2
# matplotlib.use('Qt4Agg')
sys.path.append( os.path.normpath( os.path.join('/home/panquwang/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
import labels
from labels     import trainId2label,id2label
from joblib import Parallel, delayed
import multiprocessing

def get_final_rule(saved_rule_traverse_result, performance_threshold):
    # get result
    with open(saved_rule_traverse_result, "r") as rule_traverse_result_file:
        all_rule_result_content = rule_traverse_result_file.readlines()
    all_rule_result_content = [x.strip('\n') for x in all_rule_result_content]

    # select working rules
    selected_rule_set = []
    for current_rule in all_rule_result_content:
        current_rule_split = current_rule.split('\t')
        current_rule_performance = current_rule_split[traverse_list_length + 1]
        current_rule_all_category_mean_performance = np.mean([float(i) for i in current_rule_split[4:-3]])
        if current_rule_all_category_mean_performance > performance_threshold:
            selected_rule_set.append((current_rule_split[:4]))

    for index_rule, rule in enumerate(selected_rule_set):
        for index_item, item in enumerate(rule):
            selected_rule_set[index_rule][index_item] = float(selected_rule_set[index_rule][index_item])

    final_selected_rule_set = ([], [])
    for index, selected_rule in enumerate(selected_rule_set):
        final_selected_rule_set[0].append(selected_rule[:3])
        final_selected_rule_set[1].append(selected_rule[-1])

    return final_selected_rule_set

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

def predict(index,random_list,gt_files,folder_files,final_selected_rule_set,original_image_files,result_location,is_test_lower_bound,is_use_neighbor,traverse_category_list,current_set):
# def predict(random_list, gt_files, folder_files, final_selected_rule_set, original_image_files,
#                 result_location, is_test_lower_bound, is_use_neighbor, traverse_category_list, current_set):

    img_width=2048
    img_height=1024

    # iterate through all images
    # for index in range(len(superpixel_data)):
    original_image = cv2.imread(original_image_files[index])
    file_name = original_image_files[index].split('/')[-1][:-4]+'.png'
    print str(index) + ' ' + file_name

    # gather prediction maps, form multi-layer maps
    current_all_layer_values = np.zeros((img_height, img_width, len(folder_files)))
    for key, value in folder.iteritems():
        current_layer_value = cv2.imread(folder_files[key][index], 0)
        current_all_layer_values[:, :, key - 1]=convert_label_to_trainid(current_layer_value)

    final_map=np.ones((img_height, img_width))*(255)

    # pixel level rule application
    for index_row, row in enumerate(current_all_layer_values):
        # print index_row
        for index_col, col in enumerate(row):
            value=copy.deepcopy(current_all_layer_values[index_row][index_col])

            # set all other labels to ignore label
            for index_single_value in range(len(value)):
                if not (value[index_single_value] in traverse_category_list[:-1]):
                    value[index_single_value]=traverse_category_list[-1]

            if current_set=="345":
                # apply the hard-coded primming rule
                if value[0] == 3:
                    value[0] = 255
                if value[1] == 5 or value[1] == 4:
                    value[1] = 255
                if value[2] == 4 or value[2] == 5:
                    value[2] = 255
                if value[3] == 5:
                    value[3] = 255
                if value[4] == 3 or value[4] == 5:
                    value[4] = 255

            elif current_set == "679":
                if value[2] == 6 or value[2] == 7 or value[2] == 9:
                    value[2] = 255
                if value[3] == 6 or value[3] == 7:
                    value[3] = 255
                if value[4] == 6 or value[4] == 7:
                    value[4] = 255

            elif current_set == "141516":
                # apply the hard-coded primming rule to avoid bug such as (1,2,3,4) not treated as (255,2,3,4)
                if value[0] == 14:
                    value[0] = 255
                if value[1] == 14 or value[1] == 15 or value[1] == 16:
                    value[1] = 255
                if value[2] == 16:
                    value[2] = 255

            # if current pixel meets the rule
            if value.tolist() in final_selected_rule_set[0]:
                selected_rule_index = final_selected_rule_set[0].index(value.tolist())
                # if this label belongs to the 4 big object categories
                if final_selected_rule_set[1][selected_rule_index] != 255:
                    final_map[index_row][index_col] = final_selected_rule_set[1][selected_rule_index]
                else:  # if this label belongs to other categories
                    index_255 = final_selected_rule_set[0][selected_rule_index].index(255)
                    final_map[index_row][index_col] = current_all_layer_values[index_row][index_col][index_255]
            # if current pixel does not meet the rule
            else:
                final_map[index_row][index_col] = current_all_layer_values[index_row][index_col][0]


    # save score
    final_map_saved=copy.deepcopy(final_map)
    score=convert_trainid_to_label(final_map)
    cv2.imwrite(os.path.join(result_location,'score',file_name),score)

    # save visualization
    # original image
    concat_img = Image.new('RGB', (img_width * 3, img_height))
    concat_img.paste(Image.fromarray(original_image[:, :, [2, 1, 0]]).convert('RGB'), (0, 0))
    # ground truth
    gt_img = Image.open(gt_files[index])
    concat_img.paste(gt_img, (img_width, 0))
    # prediction
    final_map_saved = final_map_saved.astype(np.uint8)
    result_img = Image.fromarray(final_map_saved).convert('P')
    palette = get_palette()
    result_img.putpalette(palette)
    # concat_img.paste(result_img, (2048*2,0))
    concat_img.paste(result_img.convert('RGB'), (img_width * 2, 0))
    concat_img.save(os.path.join(result_location, 'visualization', file_name))


def get_single_set_rules(current_set):
    all_possible_rule_list = []

    if current_set=='345':
        all_possible_rule_list.append(([2, 3, 3, 2], 3))


    elif current_set == '679':
        all_possible_rule_list.append(([255, 255, 9, 9], 9))

    elif current_set == '141516':
        all_possible_rule_list.append(([255, 15, 16, 15], 16))


    final_selected_rule_set = ([], [])
    for index, content in enumerate(all_possible_rule_list):
        final_selected_rule_set[0].append(all_possible_rule_list[index][0])
        final_selected_rule_set[1].append(all_possible_rule_list[index][1])

    return final_selected_rule_set


if __name__ == '__main__':
    dataset='val'
    current_set='141516'

    is_test_lower_bound=0
    is_use_neighbor=0
    is_get_subset_category_data=0
    is_use_list=0


    original_image_folder = '/mnt/scratch/panqu/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'+dataset+'_for_test/'
    original_image_files=glob.glob(os.path.join(original_image_folder,"*.png"))
    original_image_files.sort()

    gt_folder = '/mnt/scratch/panqu/Dataset/CityScapes/gtFine/'+dataset+'_for_test/'
    gt_files=glob.glob(os.path.join(gt_folder,"*gtFine_color.png"))
    gt_files.sort()


    # use 207 validation subfolder
    folder = {}
    # base: resnet 152
    folder[1] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_resnet_152_'+dataset+'_crf_test/')
    # deconv 1.25
    folder[2] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_deconv_scale125_crf_' + dataset + '_test')
    # scale 0.5
    folder[3] = os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/', dataset, dataset + '-epoch-39-CRF-050-test')
    # wild atrous
    folder[4] = os.path.join('/mnt/scratch/pengfei/crf_results/yenet_asppp_wild_atrous_epoch20_crf_' + dataset + '_test')
    # coarse
    folder[5] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_select_coarse_fine_epoch16_crf_' + dataset + '_test')

    folder_files={}
    for key,value in folder.iteritems():
        folder_files[key]=glob.glob(os.path.join(value,'*.png'))
        folder_files[key].sort()




    print "start to predict..."
    random_list=range(0,len(original_image_files))

    if current_set=='345':
        traverse_category_list=[3,4,5,255] # you only want to explore four categories (255 means all others)
    elif current_set == '679':
        traverse_category_list=[6,7,9,255] # you only want to explore four categories (255 means all others)
    elif current_set == '141516':
        traverse_category_list=[14,15,16,255] # you only want to explore four categories (255 means all others)

    # traverse_list_length=3 # you have three layers for ensemble
    # traverse_category_list=[3,14,15,16,255] # you only want to explore four categories (255 means all others)
    # # This is used in prediction.
    # performance_threshold=0.75368
    # saved_rule_traverse_result='/home/panquwang/adas-segmentation-cityscape/test/rule_traverse_result_file_with_purity.txt'
    # final_selected_rule_set=get_final_rule(saved_rule_traverse_result,performance_threshold)


    final_selected_rule_set=get_single_set_rules(current_set)

    # prediction
    result_location = os.path.join('/mnt/scratch/panqu/SLIC/prediction_result/four_layers_rule_traverse_use_coarse/', dataset,'all_selected_rules_pixel_level_'+current_set)
    if not os.path.exists(result_location):
        os.makedirs(result_location)
        os.makedirs(os.path.join(result_location, 'score'))
        os.makedirs(os.path.join(result_location, 'visualization'))

    num_cores = multiprocessing.cpu_count()
    range_i = range(0, len(original_image_files))

    Parallel(n_jobs=num_cores)(
        delayed(predict)(i, random_list,gt_files,folder_files,final_selected_rule_set,original_image_files,result_location,is_test_lower_bound,is_use_neighbor,traverse_category_list,current_set) for i in
        range_i)

    # predict(random_list,gt_files,folder_files,final_selected_rule_set,original_image_files,result_location,is_test_lower_bound,is_use_neighbor,traverse_category_list,current_set)