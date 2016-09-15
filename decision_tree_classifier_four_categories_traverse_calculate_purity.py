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


def get_quality(superpixel_label,current_all_layer_values, index_superpixel):
    quality=False
    # ground-truth of current superpixel
    predict_label = superpixel_label[superpixel_label == index_superpixel]

    # superpixel too small
    if len(predict_label)<=1:
        return False,0,[]

    # if prediction label does not agree with each other uniformly in this area. (Set agreement threshold to 0.5 by default)
    agreement_threshold=0.5
    predict_label_count=Counter(predict_label).most_common()
    predict_label_consistency_rate=float(predict_label_count[0][1])/len(predict_label)

    if predict_label_consistency_rate<agreement_threshold:
        return False,predict_label_consistency_rate,predict_label_count

    # otherwise pass the quality test
    return True,predict_label_consistency_rate,predict_label_count

def predict(random_list,superpixel_data,gt_files,folder_files,final_selected_rule_set,original_image_files,result_location,is_test_lower_bound,is_use_neighbor,traverse_category_list):

    img_width=2048
    img_height=1024

    # iterate through all images
    # for index in range(len(superpixel_data)):
    for index in random_list:
        file_name = superpixel_data[index].split('/')[-1][:-4]+'.png'
        print str(index) + ' ' + file_name
        current_superpixel_data = cPickle.load(open(superpixel_data[index], "rb"))
        current_gt = cv2.imread(gt_files[index], 0)
        original_image = cv2.imread(original_image_files[index])

        # gather prediction maps, form multi-layer maps
        current_all_layer_values = np.zeros((img_height, img_width, len(folder_files)))
        for key, value in folder.iteritems():
            current_layer_value = cv2.imread(folder_files[key][index], 0)
            current_all_layer_values[:, :, key - 1]=convert_label_to_trainid(current_layer_value)

        # iterate through superpixel data of current image
        superpixel_label = current_superpixel_data[2]
        num_superpixels = np.max(superpixel_label) + 1

        # statistics
        each_label_size = []
        for index_superpixel in range(int(num_superpixels)):
            length = len(superpixel_label[superpixel_label == index_superpixel])
            each_label_size.append(length)


        final_map=np.ones((img_height, img_width))*(int(num_superpixels)+10)
        superpixel_feature_set=[]
        superpixel_index_set=[]
        superpixel_categorical_label=[]
        for index_superpixel in range(int(num_superpixels)):
            # # decide the quality of current superpixel. Note: this is actually a test phase, so there should not be a GT!
            quality,gt_label_consistency_rate,predict_label_count = get_quality(superpixel_label,current_all_layer_values, index_superpixel)
            if not quality:
                # why bother to predict those regions? set to ignore label.
                continue

            # extract a 100 dimensional feature for current super pixel
            feature, label, categorical_label=get_feature_single_superpixel(superpixel_label,current_all_layer_values, index_superpixel,gt_label_consistency_rate,predict_label_count)

            # save to a single set to increase processing speed
            feature_get=[feature[i] for i in range(38,58)+range(59,79)+range(80,100)]

            # feature_get=feature[38:]
            superpixel_feature_set.append(feature_get)
            superpixel_index_set.append(index_superpixel)
            superpixel_categorical_label.append(categorical_label)

        superpixel_categorical_label_copy=copy.deepcopy(superpixel_categorical_label)
        # then assign label to all superpixels
        for predicted_index,superpixel_index in enumerate(superpixel_index_set):
            current_categorical_labels=superpixel_categorical_label_copy[predicted_index]
            # set all other labels to ignore label
            for current_categorical_label_index, current_categorical_label in enumerate(current_categorical_labels):
                if current_categorical_label not in traverse_category_list[:-1]:
                    current_categorical_labels[current_categorical_label_index]=traverse_category_list[-1]


            # if current superpixel meets the rule

            if current_categorical_labels in final_selected_rule_set[0]:
            # if current_categorical_labels==final_selected_rule_set[0]:
                selected_rule_index=final_selected_rule_set[0].index(current_categorical_labels)

                if final_selected_rule_set[1][selected_rule_index]!=255: # if this label belongs to the 4 big object categories
                    final_map[superpixel_label == superpixel_index] = final_selected_rule_set[1][selected_rule_index]
                else: # if this label belongs to other categories
                    index_255=final_selected_rule_set[0][selected_rule_index].index(255)
                    final_map[superpixel_label == superpixel_index] = superpixel_categorical_label[predicted_index][index_255]
            # if current superpixel does not meet the rule
            else:
                final_map[superpixel_label == superpixel_index] = superpixel_categorical_label[predicted_index][0]


        final_map[final_map>int(num_superpixels)]=255

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

if __name__ == '__main__':
    dataset = 'val'

    is_calculate_purity = 0
    is_load_purity_result=1

    original_image_folder = '/mnt/scratch/panqu/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/' + dataset + '_for_traverse/'
    original_image_files = glob.glob(os.path.join(original_image_folder, "*.png"))
    original_image_files.sort()

    gt_folder = '/mnt/scratch/panqu/Dataset/CityScapes/gtFine/' + dataset + '_for_traverse/'
    gt_files = glob.glob(os.path.join(gt_folder, "*gtFine_labelTrainIds.png"))
    gt_files.sort()

    superpixel_result_folder = '/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_' + dataset + '_subset/data/'
    superpixel_data = glob.glob(os.path.join(superpixel_result_folder, '*.dat'))
    superpixel_data.sort()

    # prediction for validation set
    # folder={}
    # # base
    # folder[1]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/',dataset, dataset+'-epoch-35-CRF', 'score')
    # # truck, wall
    # folder[2]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/',dataset, dataset+'-epoch-39-CRF-050', 'score')
    # # bus, train
    # folder[3]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_atrous16_epoch_33/', dataset, dataset+'-epoch-33-CRF', 'score')

    # use 150 validation subfolder
    folder = {}
    # base:
    folder[1] = os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/', dataset,
                             dataset + '-epoch-35-CRF_for_traverse', 'score')
    # scale 05
    folder[2] = os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/', dataset,
                             dataset + '-epoch-39-CRF-050_for_traverse', 'score')
    # wild atrous
    folder[3] = os.path.join(
        '/mnt/scratch/pengfei/crf_results/yenet_asppp_wild_atrous_epoch16_' + dataset + '_subset_crf', 'score')
    # deconv
    folder[4] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_deconv_epoch30_' + dataset + '_subset_crf',
                             'score')

    folder_files = {}
    for key, value in folder.iteritems():
        folder_files[key] = glob.glob(os.path.join(value, '*.png'))
        folder_files[key].sort()

    print "start to predict..."

    traverse_list_length = 4  # you have three layers for ensemble
    random_list = range(0, 233)

    if is_calculate_purity:
        traverse_category_list_2345 = [2, 3, 4, 5, 255]  # you only want to explore several categories (255 means all others)
        get_stats_2345(traverse_category_list_2345,folder_files,gt_files)

        traverse_category_list_6789 = [6, 7, 8, 9, 255]  # you only want to explore several categories (255 means all others)
        get_stats_6789(traverse_category_list_6789,folder_files,gt_files)

        traverse_category_list_13141516 = [13, 14, 15, 16, 255]  # you only want to explore several categories (255 means all others)
        get_stats_13141516(traverse_category_list_13141516,folder_files,gt_files)

    if is_load_purity_result:
        purity_2345 = cPickle.load(open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_2345.dat'), "rb"))
        purity_6789 = cPickle.load(open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_6789.dat'), "rb"))
        purity_13141516 = cPickle.load(open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_13141516.dat'), "rb"))

        print "calculating stats..."
        purity_2345_final = calculate_purity(purity_2345)
        purity_6789_final = calculate_purity(purity_6789)
        purity_13141516_final = calculate_purity(purity_13141516)

        cPickle.dump((purity_2345_final,purity_6789_final,purity_13141516_final), open(os.path.join('/home/panquwang/SLIC_cityscapes/', 'purity_final.dat'), "w+"))

        # saved_rule_traverse_result='/home/panquwang/adas-segmentation-cityscape/test/rule_traverse_result_file_with_purity.txt'
    #
    #
    # # prediction
    # result_location = os.path.join('/mnt/scratch/panqu/SLIC/prediction_result/four_cats_rule_traverse/', dataset,'all_selected_rules')
    # if not os.path.exists(result_location):
    #     os.makedirs(result_location)
    #     os.makedirs(os.path.join(result_location, 'score'))
    #     os.makedirs(os.path.join(result_location, 'visualization'))



