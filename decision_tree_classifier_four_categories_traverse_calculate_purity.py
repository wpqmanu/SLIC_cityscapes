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

def get_stats_2345(traverse_category_list,folder_files,gt_files):
    # go through all training files
    C=Counter()
    for index in range(len(gt_files)):
        print index
        four_layers_combined=np.ones((1024,2048,len(folder_files)+1))*256
        # for each file, go through every layer
        for key, value in folder_files.iteritems():
            current_layer=cv2.imread(value[index],0)
            current_layer_trainId=convert_label_to_trainid(current_layer)
            current_layer_trainId[np.where(np.logical_or(current_layer_trainId<traverse_category_list[0], current_layer_trainId>traverse_category_list[-2]))]=255
            # apply layer specific rules
            if key==1:
                current_layer_trainId[current_layer_trainId==3]=255
            elif key == 2:
                current_layer_trainId[current_layer_trainId == 2] = 255
                current_layer_trainId[current_layer_trainId == 5] = 255
            elif key==3:
                current_layer_trainId[current_layer_trainId==5]=255
            elif key==4:
                current_layer_trainId[current_layer_trainId==3]=255
            four_layers_combined[:,:,key-1]=current_layer_trainId
        gt=cv2.imread(gt_files[index],0)
        gt[np.where(np.logical_or(gt < traverse_category_list[0],gt > traverse_category_list[-2]))] = 255
        four_layers_combined[:,:,-1]=gt

        # statistics
        print "find unique rows..."
        four_layers_combined=np.transpose(four_layers_combined,(2,0,1))
        four_layers_combined=four_layers_combined.reshape((len(folder_files)+1,1024*2048))
        four_layers_combined=np.transpose(four_layers_combined,(1,0))
        new_array=[tuple(row) for row in four_layers_combined]
        C=C+Counter(new_array)

    cPickle.dump(C, open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_2345.dat'), "w+"))

def get_stats_6789(traverse_category_list,folder_files,gt_files):
    # go through all training files
    C=Counter()
    for index in range(len(gt_files)):
        print index
        four_layers_combined=np.ones((1024,2048,len(folder_files)+1))*256
        # for each file, go through every layer
        for key, value in folder_files.iteritems():
            current_layer=cv2.imread(value[index],0)
            current_layer_trainId=convert_label_to_trainid(current_layer)
            current_layer_trainId[np.where(np.logical_or(current_layer_trainId<traverse_category_list[0], current_layer_trainId>traverse_category_list[-2]))]=255
            # apply layer specific rules
            if key == 2:
                current_layer_trainId[current_layer_trainId == 6] = 255
                current_layer_trainId[current_layer_trainId == 7] = 255
                current_layer_trainId[current_layer_trainId == 8] = 255
                current_layer_trainId[current_layer_trainId == 9] = 255
            elif key==3:
                current_layer_trainId[current_layer_trainId == 6] = 255
                current_layer_trainId[current_layer_trainId == 7] = 255
            four_layers_combined[:,:,key-1]=current_layer_trainId
        gt=cv2.imread(gt_files[index],0)
        gt[np.where(np.logical_or(gt < traverse_category_list[0],gt > traverse_category_list[-2]))] = 255
        four_layers_combined[:,:,-1]=gt

        # statistics
        print "find unique rows..."
        four_layers_combined=np.transpose(four_layers_combined,(2,0,1))
        four_layers_combined=four_layers_combined.reshape((len(folder_files)+1,1024*2048))
        four_layers_combined=np.transpose(four_layers_combined,(1,0))
        new_array=[tuple(row) for row in four_layers_combined]
        C=C+Counter(new_array)

    cPickle.dump(C, open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_6789.dat'), "w+"))

def get_stats_13141516(traverse_category_list,folder_files,gt_files):
    # go through all training files
    C=Counter()
    for index in range(len(gt_files)):
        print index
        four_layers_combined=np.ones((1024,2048,len(folder_files)+1))*256
        # for each file, go through every layer
        for key, value in folder_files.iteritems():
            current_layer=cv2.imread(value[index],0)
            current_layer_trainId=convert_label_to_trainid(current_layer)
            current_layer_trainId[np.where(np.logical_or(current_layer_trainId<traverse_category_list[0], current_layer_trainId>traverse_category_list[-2]))]=255
            # apply layer specific rules
            if key==1:
                current_layer_trainId[current_layer_trainId == 14] = 255
                current_layer_trainId[current_layer_trainId == 15] = 255
                current_layer_trainId[current_layer_trainId == 16] = 255
            elif key == 2:
                current_layer_trainId[current_layer_trainId == 13] = 255
            elif key==3:
                current_layer_trainId[current_layer_trainId == 13] = 255
            elif key==4:
                current_layer_trainId[current_layer_trainId == 13] = 255
                current_layer_trainId[current_layer_trainId == 14] = 255
            four_layers_combined[:,:,key-1]=current_layer_trainId
        gt=cv2.imread(gt_files[index],0)
        gt[np.where(np.logical_or(gt < traverse_category_list[0],gt > traverse_category_list[-2]))] = 255
        four_layers_combined[:,:,-1]=gt

        # statistics
        print "find unique rows..."
        four_layers_combined=np.transpose(four_layers_combined,(2,0,1))
        four_layers_combined=four_layers_combined.reshape((len(folder_files)+1,1024*2048))
        four_layers_combined=np.transpose(four_layers_combined,(1,0))
        new_array=[tuple(row) for row in four_layers_combined]
        C=C+Counter(new_array)

    cPickle.dump(C, open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_13141516.dat'), "w+"))

def get_stats_345(traverse_category_list,folder_files,gt_files):
    # go through all training files
    C=Counter()
    for index in range(len(gt_files)):
        print index
        four_layers_combined=np.ones((1024,2048,len(folder_files)+1))*256
        # for each file, go through every layer
        for key, value in folder_files.iteritems():
            current_layer=cv2.imread(value[index],0)
            current_layer_trainId=convert_label_to_trainid(current_layer)
            current_layer_trainId[np.where(np.logical_or(current_layer_trainId<traverse_category_list[0], current_layer_trainId>traverse_category_list[-2]))]=255
            # apply layer specific rules
            if key==1:
                current_layer_trainId[current_layer_trainId==3]=255
            elif key == 2:
                current_layer_trainId[current_layer_trainId == 3] = 255
                current_layer_trainId[current_layer_trainId == 4] = 255
            elif key==3:
                current_layer_trainId[current_layer_trainId==4]=255
                current_layer_trainId[current_layer_trainId==5]=255
            elif key==4:
                current_layer_trainId[current_layer_trainId==5]=255
            elif key==5:
                current_layer_trainId[current_layer_trainId==3]=255
                current_layer_trainId[current_layer_trainId==5]=255
            four_layers_combined[:,:,key-1]=current_layer_trainId
        gt=cv2.imread(gt_files[index],0)
        gt[np.where(np.logical_or(gt < traverse_category_list[0],gt > traverse_category_list[-2]))] = 255
        four_layers_combined[:,:,-1]=gt

        # statistics
        print "find unique rows..."
        four_layers_combined=np.transpose(four_layers_combined,(2,0,1))
        four_layers_combined=four_layers_combined.reshape((len(folder_files)+1,1024*2048))
        four_layers_combined=np.transpose(four_layers_combined,(1,0))
        new_array=[tuple(row) for row in four_layers_combined]
        C=C+Counter(new_array)

    cPickle.dump(C, open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_345_use_coarse.dat'), "w+"))

def get_stats_679(traverse_category_list,folder_files,gt_files):
    # go through all training files
    C=Counter()
    for index in range(len(gt_files)):
        print index
        four_layers_combined=np.ones((1024,2048,len(folder_files)+1))*256
        # for each file, go through every layer
        for key, value in folder_files.iteritems():
            current_layer=cv2.imread(value[index],0)
            current_layer_trainId=convert_label_to_trainid(current_layer)
            current_layer_trainId[np.where(np.logical_or(current_layer_trainId<traverse_category_list[0], current_layer_trainId>traverse_category_list[-2]))]=255
            current_layer_trainId[np.where(current_layer_trainId==8)]=255

            # apply layer specific rules
            if key==3:
                current_layer_trainId[current_layer_trainId==6]=255
                current_layer_trainId[current_layer_trainId==7]=255
                current_layer_trainId[current_layer_trainId==9]=255
            elif key==4:
                current_layer_trainId[current_layer_trainId==6]=255
                current_layer_trainId[current_layer_trainId==7]=255
            elif key==5:
                current_layer_trainId[current_layer_trainId==6]=255
                current_layer_trainId[current_layer_trainId==7]=255
            four_layers_combined[:,:,key-1]=current_layer_trainId
        gt=cv2.imread(gt_files[index],0)
        gt[np.where(np.logical_or(gt < traverse_category_list[0],gt > traverse_category_list[-2]))] = 255
        gt[np.where(gt==8)]=255
        four_layers_combined[:,:,-1]=gt

        # statistics
        print "find unique rows..."
        four_layers_combined=np.transpose(four_layers_combined,(2,0,1))
        four_layers_combined=four_layers_combined.reshape((len(folder_files)+1,1024*2048))
        four_layers_combined=np.transpose(four_layers_combined,(1,0))
        new_array=[tuple(row) for row in four_layers_combined]
        C=C+Counter(new_array)

    cPickle.dump(C, open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_679_use_coarse.dat'), "w+"))

def get_stats_141516(traverse_category_list,folder_files,gt_files):
    # go through all training files
    C=Counter()
    for index in range(len(gt_files)):
        print index
        four_layers_combined=np.ones((1024,2048,len(folder_files)+1))*256
        # for each file, go through every layer
        for key, value in folder_files.iteritems():
            current_layer=cv2.imread(value[index],0)
            current_layer_trainId=convert_label_to_trainid(current_layer)
            current_layer_trainId[np.where(np.logical_or(current_layer_trainId<traverse_category_list[0], current_layer_trainId>traverse_category_list[-2]))]=255
            # apply layer specific rules
            if key==1:
                current_layer_trainId[current_layer_trainId == 14] = 255
            elif key == 2:
                current_layer_trainId[current_layer_trainId == 14] = 255
                current_layer_trainId[current_layer_trainId == 15] = 255
                current_layer_trainId[current_layer_trainId == 16] = 255
            elif key==3:
                current_layer_trainId[current_layer_trainId == 16] = 255

            four_layers_combined[:,:,key-1]=current_layer_trainId
        gt=cv2.imread(gt_files[index],0)
        gt[np.where(np.logical_or(gt < traverse_category_list[0],gt > traverse_category_list[-2]))] = 255
        four_layers_combined[:,:,-1]=gt

        # statistics
        print "find unique rows..."
        four_layers_combined=np.transpose(four_layers_combined,(2,0,1))
        four_layers_combined=four_layers_combined.reshape((len(folder_files)+1,1024*2048))
        four_layers_combined=np.transpose(four_layers_combined,(1,0))
        new_array=[tuple(row) for row in four_layers_combined]
        C=C+Counter(new_array)

    cPickle.dump(C, open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_141516_use_coarse.dat'), "w+"))

def calculate_purity(stats):
    new_list=[]
    for key,value in stats.iteritems():
        current_list=list(key)+[value]
        new_list.append(current_list)

    new_list_copy=copy.deepcopy(new_list)
    new_list_first_4=np.array([i[:4] for i in new_list])
    for index,item in enumerate(new_list):
        first_four_value=item[:4]
        same_first_four_values=np.where(np.all(new_list_first_4==first_four_value,axis=1))
        print same_first_four_values
        total_count_same_first_four_values=[new_list[i][-1] for i in same_first_four_values[0]]
        purity_current_rule=float(item[-1])/sum(total_count_same_first_four_values)
        new_list_copy[index].append(purity_current_rule)

    return new_list_copy

def calculate_purity_with_coarse(stats):
    new_list=[]
    for key,value in stats.iteritems():
        current_list=list(key)+[value]
        new_list.append(current_list)

    new_list_copy=copy.deepcopy(new_list)
    new_list_first_5=np.array([i[:5] for i in new_list])
    for index,item in enumerate(new_list):
        first_five_value=item[:5]
        same_first_five_values=np.where(np.all(new_list_first_5==first_five_value,axis=1))
        print same_first_five_values
        total_count_same_first_five_values=[new_list[i][-1] for i in same_first_five_values[0]]
        purity_current_rule=float(item[-1])/sum(total_count_same_first_five_values)
        new_list_copy[index].append(purity_current_rule)

    return new_list_copy


if __name__ == '__main__':
    dataset = 'val'

    is_calculate_purity = 0
    is_load_purity_result=1

    original_image_folder = '/mnt/scratch/panqu/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'+dataset+'_for_train/'
    original_image_files=glob.glob(os.path.join(original_image_folder,"*.png"))
    original_image_files.sort()

    gt_folder = '/mnt/scratch/panqu/Dataset/CityScapes/gtFine/'+dataset+'_for_train/'
    gt_files=glob.glob(os.path.join(gt_folder,"*gtFine_labelTrainIds.png"))
    gt_files.sort()

    # use 207 validation subfolder
    folder = {}
    # base: resnet 152
    folder[1] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_resnet_152_'+dataset+'_crf_train/')
    # deconv 1.25
    folder[2] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_deconv_scale125_crf_' + dataset + '_train')
    # scale 0.5
    folder[3] = os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/', dataset, dataset + '-epoch-39-CRF-050-train')
    # wild atrous
    folder[4] = os.path.join('/mnt/scratch/pengfei/crf_results/yenet_asppp_wild_atrous_epoch20_crf_' + dataset + '_train')
    # coarse
    folder[5] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_select_coarse_fine_epoch16_crf_' + dataset + '_train')

    folder_files={}
    for key,value in folder.iteritems():
        folder_files[key]=glob.glob(os.path.join(value,'*.png'))
        folder_files[key].sort()


    print "start to predict..."

    traverse_list_length = 5

    if is_calculate_purity:
        traverse_category_list_345 = [3, 4, 5, 255]  # you only want to explore several categories (255 means all others)
        get_stats_345(traverse_category_list_345,folder_files,gt_files)

        traverse_category_list_679 = [6, 7, 9, 255]  # you only want to explore several categories (255 means all others)
        get_stats_679(traverse_category_list_679,folder_files,gt_files)

        traverse_category_list_141516 = [14, 15, 16, 255]  # you only want to explore several categories (255 means all others)
        get_stats_141516(traverse_category_list_141516,folder_files,gt_files)

    if is_load_purity_result:
        purity_345 = cPickle.load(open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_345_use_coarse.dat'), "rb"))
        purity_679 = cPickle.load(open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_679_use_coarse.dat'), "rb"))
        purity_141516 = cPickle.load(open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_141516_use_coarse.dat'), "rb"))

        print "calculating stats..."
        purity_345_final = calculate_purity_with_coarse(purity_345)
        purity_679_final = calculate_purity_with_coarse(purity_679)
        purity_141516_final = calculate_purity_with_coarse(purity_141516)

        cPickle.dump((purity_345_final,purity_679_final,purity_141516_final), open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_final_use_coarse.dat'), "w+"))

        # saved_rule_traverse_result='/home/panquwang/adas-segmentation-cityscape/test/rule_traverse_result_file_with_purity.txt'
    #
    #
    # # prediction
    # result_location = os.path.join('/mnt/scratch/panqu/SLIC/prediction_result/four_cats_rule_traverse/', dataset,'all_selected_rules')
    # if not os.path.exists(result_location):
    #     os.makedirs(result_location)
    #     os.makedirs(os.path.join(result_location, 'score'))
    #     os.makedirs(os.path.join(result_location, 'visualization'))



