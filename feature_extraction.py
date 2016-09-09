#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import sys
import os.path
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import glob
import cPickle
sys.path.insert(0,'/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
sys.path.append(os.path.normpath(os.path.join('/mnt/scratch/panqu/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
from labels     import trainId2label,id2label
from collections import Counter

def get_palette():
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
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
    unique_values_in_array = unique_values_in_array[::-1]
    for unique_value in unique_values_in_array:
        converted_value = id2label[unique_value].trainId
        current_layer_value[current_layer_value == unique_value] = converted_value
    return  current_layer_value

def get_quality(superpixel_label,current_all_layer_values, current_gt, index_superpixel):
    quality=False
    # ground-truth of current superpixel
    gt_label = current_gt[superpixel_label == index_superpixel]
    predict_label = superpixel_label[superpixel_label == index_superpixel]

    # superpixel too small
    if len(gt_label)<=10:
        return False,0,[]

    # superpixel contains lots of ignore region except road
    if len(gt_label)<100000 and (float(np.sum(gt_label==255))/len(gt_label))>0.75:
        return False,0,[]

    gt_label=gt_label[gt_label!=255]
    # if gt_label does not agree with each other uniformly in this area. (Set agreement threshold to 0.9 by default)
    agreement_threshold=0.9
    gt_label_count=Counter(gt_label).most_common()
    if len(gt_label_count)>0:
        gt_label_consistency_rate=float(gt_label_count[0][1])/len(gt_label)
    else:
        gt_label_consistency_rate=0
    if gt_label_consistency_rate<agreement_threshold:
        return False,gt_label_consistency_rate,gt_label_count

    agreement_threshold = 0.5
    predict_label_count = Counter(predict_label).most_common()
    predict_label_consistency_rate = float(predict_label_count[0][1]) / len(predict_label)

    if predict_label_consistency_rate < agreement_threshold:
        return False, predict_label_consistency_rate, predict_label_count


    # otherwise pass the quality test
    return True,predict_label_consistency_rate,gt_label_count


def get_feature_single_superpixel(superpixel_label,current_all_layer_values, index_superpixel,label_consistency_rate,gt_label_count):
    label=gt_label_count[0][0]
    binary_mask=(superpixel_label == index_superpixel).astype(np.uint8)

    feature=[]

    #feature dimension 0: label_consistency_rate
    feature.extend([label_consistency_rate])

    # feature dimension 1: area
    contours, hierarchy = cv2.findContours(binary_mask, 0, 2)
    contours_with_holes, hierarchy_with_holes = cv2.findContours(binary_mask, 1, 2)
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    feature.extend([area])

    # feature dimension 2: perimeter
    perimeter = cv2.arcLength(cnt, True)
    feature.extend([perimeter])

    # feature dimension 3,4: centroid
    M = cv2.moments(cnt)
    cx = int(M['m10'] / (M['m00']+1e-5))
    cy = int(M['m01'] / (M['m00']+1e-5))
    feature.extend([cx,cy])

    # feature dimension 5: convexity
    convexity=int(cv2.isContourConvex(cnt))
    feature.extend([convexity])

    # feature dimension 6,7,8,9,10: minAreaRectangle centerx, centery, width, height, angle of rotation
    rect = cv2.minAreaRect(cnt)
    feature.extend([rect[0][0],rect[0][1],rect[1][0],rect[1][1],rect[2]])

    # feature dimension  11,12,13: minEnclosingCircle centerx, centery, radius
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    feature.extend([int(x),int(y),radius])

    # feature dimension 14: aspect ratio
    x_rect,y_rect,w,h=cv2.boundingRect(cnt)
    aspect_ratio=float(w)/h
    feature.extend([aspect_ratio])

    # feature dimension 15: extent
    rect_area = w * h
    extent = float(area) / (rect_area+1e-5)
    feature.extend([extent])

    # feature dimension 16, 17: convex hull area and solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / (hull_area+1e-5)
    feature.extend([hull_area,solidity])

    # feature dimension 18: equivalent diameter
    equi_diameter = np.sqrt(4 * area / np.pi)
    feature.extend([equi_diameter])

    # feature dimension 19, 20, 21: major axis length, minor axis length, orientation
    if cnt.shape[0]<=5:
        feature.extend([0, 0, 0])
    else:
        (x_ecclipse, y_ecclipse), (MA, ma), angle = cv2.fitEllipse(cnt)
        feature.extend([MA, ma, angle])

    # feature dimension 22: Euler number
    Euler_number=len(contours)-len(contours_with_holes)
    feature.extend([Euler_number])

    # feature dimension 23, 24, 25, 26, 27, 28, 29, 30: extreme points
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    feature.extend([leftmost[0], bottommost[1], rightmost[0], rightmost[1], topmost[0], topmost[1], bottommost[0], bottommost[1]])

    # feature dimension 31, 32, 33, 34, 35, 36, 37: Hu moments
    Hu_moments=cv2.HuMoments(M).flatten()
    feature.extend(Hu_moments)

    categorical_label=[]
    # feature dimension 38-57, 58, 59-78, 79, 80-99, 100: prediction of our models and consistency
    for layer_index in range(current_all_layer_values.shape[2]):
        current_layer_current_superpixel_label = current_all_layer_values[:,:,layer_index][superpixel_label == index_superpixel]
        current_layer_label_count = Counter(current_layer_current_superpixel_label).most_common()
        current_layer_consistency_rate = float(current_layer_label_count[0][1]) / len(current_layer_current_superpixel_label)
        # one-hot encoding
        one_hot=[0]*20
        one_hot[int(current_layer_label_count[0][0])]=1
        feature.extend(one_hot+[current_layer_consistency_rate])
        categorical_label.append(current_layer_label_count[0][0])



    return feature,label,categorical_label

def extract_features(superpixel_data,gt_files,folder_files):
    # iterate through all images
    feature_set=[]
    label_set=[]
    num_features=0
    for index in range(len(superpixel_data)):
        print str(index) + ' ' + superpixel_data[index].split('/')[-1]
        current_superpixel_data = cPickle.load(open(superpixel_data[index], "rb"))
        current_gt = cv2.imread(gt_files[index], 0)

        # gather prediction maps, form multi-layer maps
        current_all_layer_values = np.zeros((1024, 2048, len(folder_files)))
        for key, value in folder.iteritems():
            current_layer_value = cv2.imread(folder_files[key][index], 0)
            current_all_layer_values[:, :, key - 1]=convert_label_to_trainid(current_layer_value)

        # iterate through superpixel data of current image
        superpixel_label = current_superpixel_data[2]
        num_superpixels = int(np.max(superpixel_label)) + 1

        # statistics
        stat=[]
        for index_superpixel in range(num_superpixels):
            stat.append(len(superpixel_label[superpixel_label==index_superpixel]))
        stat.sort()

        for index_superpixel in range(num_superpixels):
            # decide the quality of current superpixel
            quality,gt_label_consistency_rate,gt_label_count = get_quality(superpixel_label,current_all_layer_values, current_gt, index_superpixel)

            if not quality:
                continue

            # extract a 40 dimensional feature for current super pixel
            feature, label,categorical_label=get_feature_single_superpixel(superpixel_label,current_all_layer_values, index_superpixel,gt_label_consistency_rate,gt_label_count)

            feature_set.append(feature)
            label_set.append(label)
            num_features=num_features+1
            print num_features

    return feature_set, label_set


if __name__ == '__main__':
    dataset='val'

    superpixel_result_folder='/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_'+dataset+'/data/'
    superpixel_data=glob.glob(os.path.join(superpixel_result_folder,'*.dat'))
    superpixel_data.sort()

    gt_folder = os.path.join('/home/panquwang/Dataset/CityScapes/gtFine/', dataset)
    gt_files=glob.glob(os.path.join(gt_folder,"*","*gtFine_labelTrainIds.png"))
    gt_files.sort()

    # base feature map
    folder={}
    # base
    folder[1]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/', dataset,  dataset+'-epoch-35-CRF', 'score')
    # truck, wall
    folder[2]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/', dataset, dataset+'-epoch-39-CRF-050', 'score')
    # bus, train
    folder[3]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_atrous16_epoch_33/', dataset, dataset+'-epoch-33-CRF', 'score')
    folder_files={}
    previous_key=0
    for key,value in folder.iteritems():
        folder_files[key]=glob.glob(os.path.join(value,'*.png'))
        folder_files[key].sort()

    print "start to gather features..."

    feature_set,label_set=extract_features(superpixel_data,gt_files,folder_files)

    saved_location='/mnt/scratch/panqu/SLIC/features/'
    cPickle.dump((feature_set, label_set), open(os.path.join(saved_location, 'features_'+dataset+'_100.dat'), "w+"))










