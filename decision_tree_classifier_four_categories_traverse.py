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

def get_quality(superpixel_label,current_all_layer_values, index_superpixel):
    quality=False
    # ground-truth of current superpixel
    predict_label = superpixel_label[superpixel_label == index_superpixel]

    # superpixel too small
    if len(predict_label)<=10:
        return False,0,[]

    # if prediction label does not agree with each other uniformly in this area. (Set agreement threshold to 0.5 by default)
    agreement_threshold=0.5
    predict_label_count=Counter(predict_label).most_common()
    predict_label_consistency_rate=float(predict_label_count[0][1])/len(predict_label)

    if predict_label_consistency_rate<agreement_threshold:
        return False,predict_label_consistency_rate,predict_label_count

    # otherwise pass the quality test
    return True,predict_label_consistency_rate,predict_label_count

def predict(random_list,superpixel_data,gt_files,folder_files,current_rule,original_image_files,result_location,is_test_lower_bound,is_use_neighbor,traverse_category_list):

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
            # decide the quality of current superpixel. Note: this is actually a test phase, so there should not be a GT!
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
        # then assign label to all superpixel
        for predicted_index,superpixel_index in enumerate(superpixel_index_set):
            current_categorical_labels=superpixel_categorical_label_copy[predicted_index]
            # set all other labels to ignore label
            for current_categorical_label_index, current_categorical_label in enumerate(current_categorical_labels):
                if current_categorical_label not in traverse_category_list[:-1]:
                    current_categorical_labels[current_categorical_label_index]=traverse_category_list[-1]

            # if current superpixel meets the rule
            if current_categorical_labels==current_rule[0]:
                if current_rule[1]!=255: # if this label belongs to the 4 big object categories
                    final_map[superpixel_label == superpixel_index] = current_rule[1]
                else: # if this label belongs to other categories
                    index_255=current_rule[0].index(255)
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



if __name__ == '__main__':
    dataset='val'
    is_test_lower_bound=0
    is_use_neighbor=0
    is_get_subset_category_data=0

    original_image_folder = '/home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'+dataset+'/'
    original_image_files=glob.glob(os.path.join(original_image_folder,"*","*.png"))
    original_image_files.sort()

    gt_folder = '/home/panquwang/Dataset/CityScapes/gtFine/'+dataset+'/'
    gt_files=glob.glob(os.path.join(gt_folder,"*","*gtFine_color.png"))
    gt_files.sort()

    superpixel_result_folder='/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_'+dataset+'/data/'
    superpixel_data=glob.glob(os.path.join(superpixel_result_folder,'*.dat'))
    superpixel_data.sort()


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

    print "start to predict..."

    traverse_list_length=3 # you have three layers for ensemble
    traverse_category_list=[3,14,15,16,255] # you only want to explore four categories (255 means all others)
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


    all_possible_rule_list=[]
    for first_item_in_list in traverse_category_list:
        for second_item_in_list in traverse_category_list:
            for third_item_in_list in traverse_category_list:
                current_category_list=[first_item_in_list,second_item_in_list,third_item_in_list]
                if len(set(current_category_list))==1:
                    continue
                for possible_category in np.unique(np.asarray(current_category_list)):
                    all_possible_rule_list.append((current_category_list,possible_category))



    for current_rule in all_possible_rule_list:
        result_location = os.path.join('/mnt/scratch/panqu/SLIC/prediction_result/four_cats_rule_traverse/', dataset,
                                       str(current_rule[0][0])+'_'+str(current_rule[0][1])+'_'+
                                       str(current_rule[0][2])+'_'+str(current_rule[1]))
        if not os.path.exists(result_location):
            os.makedirs(result_location)
            os.makedirs(os.path.join(result_location, 'score'))
            os.makedirs(os.path.join(result_location, 'visualization'))

        predict(random_list,superpixel_data,gt_files,folder_files,current_rule,original_image_files,result_location,is_test_lower_bound,is_use_neighbor,traverse_category_list)



